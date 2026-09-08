"""
ffmpeg_reader.py
================
Modul pembaca raw video stream (RTMP/RTSP/HTTP/HLS) menggunakan FFmpeg.
Berfungsi murni sebagai backend reader pengganti cv2.VideoCapture().

File ini TIDAK menyimpan URL atau konfigurasi CCTV apapun.
Semua URL disuplai oleh pemanggil (misal: person_detection.py).
"""

import os
import shutil
import subprocess
import numpy as np


def _get_default_ffmpeg(ffmpeg_path="ffmpeg"):
    """
    Memastikan path executable FFmpeg valid:
    1. Menggunakan custom path jika ada.
    2. Mencari di sistem PATH.
    3. Menggunakan FFmpeg binary bawaan imageio_ffmpeg jika belum ada di PATH.
    """
    if ffmpeg_path and os.path.exists(ffmpeg_path):
        return ffmpeg_path

    in_path = shutil.which(ffmpeg_path)
    if in_path:
        return in_path

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    return ffmpeg_path


def _resolve_stream_url(url):
    """
    Menyesuaikan URL stream jika diperlukan.
    Server Wowza mempublikasikan stream H.265/HEVC pada path /live/
    melalui protokol HLS (HTTP port 1935 /playlist.m3u8) karena RTMP
    tradisional tidak mendukung codec H.265.
    """
    if not isinstance(url, str):
        return url

    clean = url.strip()

    # Jika menggunakan format RTMP pada server Wowza /live/ H.265 (port 1935)
    if clean.startswith("rtmp://") and ":1935/live/" in clean:
        clean = clean.replace("rtmp://", "http://")
        if not clean.endswith("/playlist.m3u8"):
            clean = clean.rstrip("/") + "/playlist.m3u8"
        return clean

    # Jika menggunakan HTTP tapi belum ditambahkan /playlist.m3u8
    if (clean.startswith("http://") or clean.startswith("https://")) and clean.endswith(".stream"):
        clean += "/playlist.m3u8"
        return clean

    return clean


class FFmpegStreamReader:
    """
    Pengganti cv2.VideoCapture() khusus untuk stream yang codec-nya
    TIDAK didukung oleh FFmpeg bawaan opencv-python (misal H.265/HEVC).

    Cara kerja:
        1. Menjalankan FFmpeg sebagai proses terpisah (subprocess).
        2. FFmpeg diminta output berupa RAW VIDEO (piksel mentah,
           format BGR24) ke stdout, BUKAN file/re-encode.
        3. Python baca stdout itu sebagai bytes, lalu diubah jadi
           array NumPy yang bentuknya sama persis seperti frame
           yang biasa didapat dari cv2.VideoCapture().read().

    Supaya frame yang dihasilkan tetap bisa dipakai apa adanya
    oleh detector YOLO, tracker ByteTrack, cv2.imshow(), dll.
    """

    def __init__(self, stream_url, width=1920, height=1080, ffmpeg_path="ffmpeg"):
        """
        stream_url  : URL stream video (RTMP/HLS/HTTP/RTSP) dari pemanggil
        width       : lebar frame output (default 1920)
        height      : tinggi frame output (default 1080)
        ffmpeg_path : path ke executable ffmpeg, default 'ffmpeg' (auto-fallback jika belum di PATH)
        """
        self.raw_url = stream_url
        self.stream_url = _resolve_stream_url(stream_url)
        self.width = int(width)
        self.height = int(height)
        self.ffmpeg_path = _get_default_ffmpeg(ffmpeg_path)

        # Ukuran 1 frame mentah dalam bytes: width * height * 3 channel warna (BGR)
        self.frame_size_bytes = self.width * self.height * 3

        perintah = [
            self.ffmpeg_path,
            "-loglevel", "error",   # stdout bersih hanya berisi data piksel video
        ]

        # Opsi timeout jaringan sesuai protokol
        if self.stream_url.startswith("rtmp://"):
            perintah.extend(["-rw_timeout", "5000000"])  # 5 detik
        elif self.stream_url.startswith("http://") or self.stream_url.startswith("https://"):
            perintah.extend([
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5"
            ])

        perintah.extend([
            "-i", self.stream_url,
            "-an",                  # abaikan audio
            "-vf", f"scale={self.width}:{self.height}",  # paksa skala konsisten
            "-f", "rawvideo",       # format output piksel mentah
            "-pix_fmt", "bgr24",    # urutan warna OpenCV (BGR)
            "-",                    # output ke stdout pipe
        ])

        try:
            self.process = subprocess.Popen(
                perintah,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[FFmpegStreamReader] Gagal memulai FFmpeg subprocess: {e}")
            self.process = None

    def reconnect(self):
        """Restart FFmpeg subprocess jika koneksi stream terputus."""
        self.release()
        self.__init__(
            self.raw_url,
            self.width,
            self.height,
            self.ffmpeg_path,
        )

    def isOpened(self):
        """Mengecek apakah proses FFmpeg masih aktif."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def _baca_bytes_lengkap(self, jumlah_bytes):
        """Membaca byte dari stdout secara konsisten sampai terkumpul jumlah_bytes."""
        potongan_data = bytearray()

        while len(potongan_data) < jumlah_bytes:
            if not self.isOpened():
                return None

            sisa_bytes = jumlah_bytes - len(potongan_data)
            chunk = self.process.stdout.read(sisa_bytes)

            if not chunk:
                return None

            potongan_data.extend(chunk)

        return bytes(potongan_data)

    def read(self):
        """
        Membaca 1 frame dari stdout FFmpeg.
        Meniru method cv2.VideoCapture.read().
        Return: (ret, frame)
        """
        raw_bytes = self._baca_bytes_lengkap(self.frame_size_bytes)

        if raw_bytes is None:
            return False, None

        frame = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
        frame = frame.reshape((self.height, self.width, 3))

        return True, frame

    def release(self):
        """Menghentikan proses FFmpeg."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except (subprocess.TimeoutExpired, Exception):
                self.process.kill()
            self.process = None

    def get(self, prop_id):
        """Method pembantu kompatibilitas dengan cv2.VideoCapture."""
        import cv2

        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return 25.0

        return 0.0