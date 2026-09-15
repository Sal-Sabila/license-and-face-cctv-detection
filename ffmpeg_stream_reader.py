import os
import re
import sys
import shutil
import subprocess
import threading
import time
import numpy as np


def find_ffmpeg_executable(custom_path="ffmpeg"):
    """
    Mencari executable FFmpeg:
    1. custom_path jika valid
    2. PATH Windows
    3. folder Scripts Python/venv
    4. beberapa lokasi umum
    """
    if custom_path and custom_path != "ffmpeg":
        if os.path.isfile(custom_path):
            return custom_path

    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path

    candidates = [
        os.path.join(sys.prefix, "Scripts", "ffmpeg.exe"),
        os.path.join(os.path.dirname(sys.executable), "Scripts", "ffmpeg.exe"),
        r"C:\Users\LENOVO\AppData\Local\Programs\Python\Python312\Scripts\ffmpeg.exe",
        r"D:\Bimaa\Magang\CCTV\venv\Scripts\ffmpeg.exe",
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return "ffmpeg"


def normalize_stream_url(url):
    """
    Normalisasi URL stream CCTV.

    Catatan penting:
    Fungsi ini HARUS mengembalikan URL asli, bukan Markdown link.
    """
    if not isinstance(url, str):
        return url

    url = url.strip()

    # Perbaiki typo nama stream:
    # GSMasukViewLuarstream -> GSMasukViewLuar.stream
    if url.startswith("rtmp://") and url.endswith("stream"):
        url = url[:-6] + ".stream"

    # Jika menggunakan Wowza RTMP dan stream name belum berekstensi .stream,
    # tambahkan .stream.
    if "103.255.15.138:1935" in url:
        match = re.search(r"/live/([^/?#]+)", url)

        if match:
            stream_name = match.group(1)

            # Bersihkan kemungkinan suffix playlist.
            stream_name = stream_name.replace("/playlist.m3u8", "")

            if stream_name.endswith("stream") and not stream_name.endswith(".stream"):
                stream_name = stream_name[:-6] + ".stream"
            elif not stream_name.endswith(".stream"):
                stream_name += ".stream"

            # PENTING:
            # Return string URL biasa, BUKAN:
            # [http://...](http://...)
            return (
                f"http://103.255.15.138:1935/"
                f"live/{stream_name}/playlist.m3u8"
            )

    return url


class FFmpegStreamReader:
    """
    Reader FFmpeg -> raw BGR frame untuk OpenCV.

    Tujuan:
    - Tidak menggunakan cv2.VideoCapture.
    - Cocok untuk RTMP / RTSP / HLS.
    - Menangani stream CCTV yang timestamp-nya tidak monoton.
    - Tidak membiarkan stderr FFmpeg memenuhi pipe.
    - Bisa reconnect ketika FFmpeg benar-benar berhenti.
    - Interface tetap kompatibel dengan:
          ret, frame = cap.read()
          cap.isOpened()
          cap.release()
          cap.get(...)
    """

    def __init__(
        self,
        rtmp_url,
        width=1280,
        height=720,
        ffmpeg_path="ffmpeg",
        max_reconnect=3,
        reconnect_delay=1.5,
    ):
        self.width = int(width)
        self.height = int(height)
        self.original_url = str(rtmp_url).strip()
        self.rtmp_url = normalize_stream_url(self.original_url)

        self.ffmpeg_path = find_ffmpeg_executable(ffmpeg_path)

        self.max_reconnect = max(1, int(max_reconnect))
        self.reconnect_delay = max(0.2, float(reconnect_delay))

        self.error_message = None
        self.last_error = None
        self.last_stderr = ""
        self._stderr_lines = []
        self._stderr_lock = threading.Lock()
        self._stderr_thread = None

        self.process = None
        self._released = False
        self._starting = False

        self.frame_size_bytes = self.width * self.height * 3

        self._start_process()

    # ------------------------------------------------------------------
    # FFmpeg PROCESS
    # ------------------------------------------------------------------

    def _build_command(self):
        url = self.rtmp_url

        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "warning",

            # Generate/fix missing timestamps when possible.
            "-fflags", "+genpts",

            # Jangan menunggu input terlalu lama.
            "-rw_timeout", "15000000",
        ]

        if url.lower().startswith("rtsp://"):
            command.extend([
                "-rtsp_transport", "tcp",
            ])

        elif url.lower().startswith("rtmp://"):
            command.extend([
                "-rtmp_live", "live",
            ])

        elif url.lower().startswith(("http://", "https://")):
            command.extend([
                "-reconnect", "1",
                "-reconnect_at_eof", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "2",
            ])

        command.extend([
            "-i", url,

            # CCTV kita hanya membutuhkan video.
            "-an",

            # Output konsisten untuk OpenCV.
            "-vf", f"scale={self.width}:{self.height}",

            # Raw BGR langsung ke stdout.
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",

            # Jangan melakukan frame duplication/drop di output.
            "-fps_mode", "passthrough",

            "pipe:1",
        ])

        return command

    def _start_process(self):
        if self._released:
            return False

        if self.process is not None:
            return self.isOpened()

        if self._starting:
            return False

        self._starting = True

        try:
            command = self._build_command()

            print(
                "[FFMPEG] Starting:",
                " ".join(command[:-2]),
                "..."
            )

            # PENTING:
            # stderr tetap PIPE tetapi selalu dibaca thread background.
            # Jika stderr tidak dibaca, warning DTS yang sangat banyak
            # dapat memenuhi OS pipe buffer dan membuat FFmpeg berhenti
            # mengirim frame ke stdout.
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            self.error_message = None

            self._stderr_lines = []

            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                daemon=True,
                name="ffmpeg-stderr",
            )
            self._stderr_thread.start()

            time.sleep(0.15)

            if self.process.poll() is not None:
                self._capture_error()
                return False

            return True

        except Exception as exc:
            self.error_message = str(exc)
            self.last_error = str(exc)
            self.process = None

            print(f"[FFMPEG ERROR] Gagal menjalankan FFmpeg: {exc}")
            return False

        finally:
            self._starting = False

    def _drain_stderr(self):
        """
        Baca stderr FFmpeg terus-menerus di thread terpisah.

        Ini sangat penting untuk kasus CCTV Anda karena FFmpeg menghasilkan
        banyak warning:
            non monotonically increasing dts

        Warning tersebut tidak boleh memenuhi stderr pipe.
        """
        process = self.process

        if process is None or process.stderr is None:
            return

        try:
            while True:
                line = process.stderr.readline()

                if not line:
                    break

                text = line.decode("utf-8", errors="replace").strip()

                if not text:
                    continue

                with self._stderr_lock:
                    self._stderr_lines.append(text)

                    # Simpan hanya beberapa baris terakhir agar RAM stabil.
                    if len(self._stderr_lines) > 40:
                        self._stderr_lines = self._stderr_lines[-40:]

        except (OSError, ValueError):
            pass

    def _capture_error(self):
        """
        Ambil error/warning terakhir dari buffer stderr.
        """
        with self._stderr_lock:
            if self._stderr_lines:
                self.last_stderr = "\n".join(self._stderr_lines[-15:])
                self.error_message = self.last_stderr[-1500:]
                self.last_error = self.error_message

    # ------------------------------------------------------------------
    # FRAME READER
    # ------------------------------------------------------------------

    def _read_exact(self, size):
        """
        Membaca tepat 'size' bytes.

        stdout.read(size) tidak selalu menjamin seluruh frame tersedia
        dalam satu read. Karena itu kita loop sampai ukuran frame lengkap.
        """
        if self.process is None or self.process.stdout is None:
            return None

        buffer = bytearray()

        while len(buffer) < size:
            process = self.process

            if process is None:
                return None

            if process.poll() is not None:
                self._capture_error()
                return None

            remaining = size - len(buffer)

            try:
                chunk = process.stdout.read(remaining)
            except (OSError, ValueError) as exc:
                self.error_message = str(exc)
                self.last_error = str(exc)
                return None

            if not chunk:
                self._capture_error()
                return None

            buffer.extend(chunk)

        return bytes(buffer)

    def read(self):
        """
        Return:
            (True, frame)
        atau:
            (False, None)

        Jika FFmpeg mati, reader mencoba reconnect beberapa kali.
        """
        if self._released:
            return False, None

        for attempt in range(self.max_reconnect + 1):
            if self.process is None:
                if not self._start_process():
                    if attempt >= self.max_reconnect:
                        return False, None

                    time.sleep(self.reconnect_delay)
                    continue

            raw_bytes = self._read_exact(self.frame_size_bytes)

            if raw_bytes is not None:
                try:
                    frame = np.frombuffer(
                        raw_bytes,
                        dtype=np.uint8,
                    ).reshape(
                        (self.height, self.width, 3)
                    ).copy()

                    return True, frame

                except Exception as exc:
                    self.error_message = f"Frame decode error: {exc}"
                    self.last_error = self.error_message
                    return False, None

            # FFmpeg benar-benar berhenti / stdout EOF.
            self._capture_error()

            if attempt >= self.max_reconnect:
                return False, None

            print(
                f"[FFMPEG] Stream berhenti. "
                f"Reconnect {attempt + 1}/{self.max_reconnect}..."
            )

            self._restart_process()
            time.sleep(self.reconnect_delay)

        return False, None

    # ------------------------------------------------------------------
    # RECONNECT
    # ------------------------------------------------------------------

    def _restart_process(self):
        self._terminate_process()
        self.process = None

        if not self._released:
            self._start_process()

    def reconnect(self):
        """
        Public method agar kode lama yang memanggil:
            cap.reconnect()
        tetap kompatibel.
        """
        if self._released:
            return False

        print("[FFMPEG] Manual reconnect...")

        self._restart_process()

        return self.isOpened()

    # ------------------------------------------------------------------
    # OPENCV-COMPATIBLE METHODS
    # ------------------------------------------------------------------

    def isOpened(self):
        """
        Meniru cv2.VideoCapture.isOpened().
        """
        if self.process is None:
            return False

        return self.process.poll() is None

    def get(self, prop_id):
        """
        Mendukung property OpenCV yang umum digunakan.
        """
        import cv2

        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width

        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height

        if prop_id == cv2.CAP_PROP_FPS:
            return 25.0

        return 0.0

    # ------------------------------------------------------------------
    # RELEASE / CLEANUP
    # ------------------------------------------------------------------

    def _terminate_process(self):
        process = self.process

        if process is None:
            return

        try:
            if process.poll() is None:
                process.terminate()

                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)

        except (OSError, ValueError):
            pass

        finally:
            self.process = None

    def release(self):
        """
        Menghentikan FFmpeg dan membersihkan resource.
        """
        self._released = True
        self._terminate_process()

        self._stderr_thread = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
