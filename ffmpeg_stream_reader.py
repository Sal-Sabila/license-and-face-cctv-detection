import os
import re
import sys
import shutil
import subprocess
import numpy as np


def find_ffmpeg_executable(custom_path="ffmpeg"):
    """Mencari lokasi executable ffmpeg di PATH atau direktori Python."""
    if custom_path and custom_path != "ffmpeg" and os.path.exists(custom_path):
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
    for c in candidates:
        if os.path.exists(c):
            return c

    return "ffmpeg"


def normalize_stream_url(url):
    """
    Normalisasi URL stream CCTV:
    1. Memperbaiki typo seperti 'GSMasukViewLuarstream' -> 'GSMasukViewLuar.stream'
    2. Jika URL adalah RTMP dari Wowza server (103.255.15.138:1935),
       karena Wowza mengirimkan stream HEVC (H.265) lewat RTMP FLV tag 0x0c
       yang tidak didukung demuxer FLV bawaan FFmpeg (error: Video codec (c) is not implemented),
       secara otomatis alihkan ke stream HLS (m3u8) resmi dari Wowza di port yang sama (1935).
       Stream HLS ini 100% didukung FFmpeg untuk decoding H.265 secara native tanpa drop frame.
    """
    if not isinstance(url, str):
        return url

    url = url.strip()

    if "103.255.15.138:1935" in url:
        m = re.search(r'/live/([^/?#]+)', url)
        if m:
            stream_name = m.group(1).replace('/playlist.m3u8', '')
            if stream_name.endswith('stream') and not stream_name.endswith('.stream'):
                stream_name = stream_name[:-6] + '.stream'
            elif not stream_name.endswith('.stream'):
                stream_name = stream_name + '.stream'
            return f"http://103.255.15.138:1935/live/{stream_name}/playlist.m3u8"

    if url.startswith("rtmp://") and url.endswith("stream") and not url.endswith(".stream"):
        url = url[:-6] + ".stream"

    return url


class FFmpegStreamReader:
    """
    Pengganti cv2.VideoCapture() khusus untuk stream yang codec-nya
    TIDAK didukung oleh FFmpeg bawaan opencv-python (misal H.265/HEVC).
    """

    def __init__(self, rtmp_url, width, height, ffmpeg_path="ffmpeg"):
        """
        rtmp_url    : URL stream RTMP / RTSP / HLS
        width       : lebar frame target
        height      : tinggi frame target
        ffmpeg_path : path ke ffmpeg.exe, default auto-detect
        """

        self.width = width
        self.height = height
        self.rtmp_url = rtmp_url
        self.ffmpeg_path = ffmpeg_path
        self.error_message = None


        # Ukuran 1 frame mentah dalam bytes:
        # width * height * 3 channel warna (BGR), 1 byte per channel
        self.frame_size_bytes = width * height * 3

        url_str = str(self.rtmp_url).strip()
        perintah = [
            self.ffmpeg_path,
            "-loglevel", "error",   # supaya stdout FFmpeg bersih
        ]

        if url_str.startswith("rtsp://"):
            perintah.extend([
                "-rtsp_transport", "tcp",
                "-stimeout", "15000000",
            ])
        elif url_str.startswith("rtmp://"):
            perintah.extend([
                "-rtmp_live", "live",
                "-rw_timeout", "15000000",
            ])
        elif url_str.startswith("http://") or url_str.startswith("https://"):
            perintah.extend([
                "-reconnect", "1",
                "-reconnect_at_eof", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "2",
                "-rw_timeout", "15000000",
            ])

        perintah.extend([
            "-i", self.rtmp_url,

            "-an",                  # tidak perlu audio, buang saja

            # PENTING: paksa ukuran output PERSIS width x height yang kita minta
            "-vf", f"scale={width}:{height}",

            "-f", "rawvideo",       # output format: piksel mentah
            "-pix_fmt", "bgr24",    # urutan warna sama seperti OpenCV
            "-",                    # tulis hasil ke stdout
        ])

        try:
            self.process = subprocess.Popen(
                perintah,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            print(f"[FFMPEG ERROR] Gagal menjalankan proses FFmpeg: {e}")
            self.process = None

    def reconnect(self):
        """Restart FFmpeg after a remote stream EOF or network timeout."""
        self.release()
        self.__init__(
            self.rtmp_url,
            self.width,
            self.height,
            self.ffmpeg_path,
        )

    def isOpened(self):
        """
        Mengecek apakah proses FFmpeg masih hidup.
        Meniru method isOpened() milik cv2.VideoCapture.
        """

        if self.process is None:
            return False

        return self.process.poll() is None

    def _baca_bytes_lengkap(self, jumlah_bytes):
        """
        stdout.read(n) TIDAK menjamin dapat tepat n bytes sekali
        panggil -- ini disebut "short read", bisa terjadi kalau
        data dari FFmpeg belum semuanya sampai di pipe saat itu.

        Kalau ini tidak diantisipasi, byte-byte dari 1 frame akan
        tercampur/kegeser dengan frame berikutnya, menyebabkan
        gambar jadi noise/pecah (persis seperti yang terjadi).

        Makanya di sini kita loop terus membaca SAMPAI benar-benar
        terkumpul tepat sejumlah `jumlah_bytes`, baru berhenti.
        """

        potongan_data = bytearray()

        while len(potongan_data) < jumlah_bytes:

            if self.process is None or self.process.poll() is not None:
                self._capture_error()
                return None

            sisa_bytes = jumlah_bytes - len(potongan_data)

            chunk = self.process.stdout.read(sisa_bytes)

            if not chunk:

                # FFmpeg berhenti mengirim data -> stream putus
                self._capture_error()
                return None

            potongan_data.extend(chunk)

        return bytes(potongan_data)

    def _capture_error(self):
        """Menyimpan pesan FFmpeg terakhir untuk diagnosis operator."""
        if self.process is None or self.process.stderr is None:
            return
        try:
            error = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            if error:
                self.error_message = error[-500:]
        except (OSError, ValueError):
            pass

    def read(self):
        """
        Membaca 1 frame dari stdout FFmpeg.
        Meniru method read() milik cv2.VideoCapture, yaitu
        return (ret, frame) supaya bisa langsung dipakai
        tanpa ubah kode yang sudah ada.
        """

        raw_bytes = self._baca_bytes_lengkap(self.frame_size_bytes)

        if raw_bytes is None:

            # Data tidak lengkap -> stream putus / proses berhenti
            return False, None

        # .copy() penting di sini: np.frombuffer() menghasilkan array
        # READ-ONLY (cuma "menumpang" di memory buffer asli, tidak
        # punya memory sendiri). Fungsi seperti cv2.putText() atau
        # cv2.rectangle() butuh menulis LANGSUNG ke frame, jadi kalau
        # tidak di-copy dulu, akan muncul error "readonly array".
        frame = np.frombuffer(raw_bytes, dtype=np.uint8).copy()

        frame = frame.reshape((self.height, self.width, 3))

        return True, frame

    def release(self):
        """
        Menghentikan proses FFmpeg.
        Meniru method release() milik cv2.VideoCapture.
        """

        if self.process is not None:

            self.process.terminate()

            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()

            self.process = None

    def get(self, prop_id):
        """
        Method dummy supaya kode lama yang manggil
        cap.get(cv2.CAP_PROP_FRAME_WIDTH) dll tidak error.
        Cuma return nilai yang sudah kita ketahui dari ffprobe.
        """

        import cv2

        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width

        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height

        return 0