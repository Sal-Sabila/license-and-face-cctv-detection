import subprocess
import numpy as np


class FFmpegStreamReader:
    """
    Pengganti cv2.VideoCapture() khusus untuk stream yang codec-nya
    TIDAK didukung oleh FFmpeg bawaan opencv-python (misal H.265/HEVC).

    Cara kerja:
        1. Menjalankan FFmpeg (yang ter-install di sistem/PATH kalian,
           yang sudah terbukti bisa decode H.265 lewat ffprobe/ffplay)
           sebagai proses terpisah (subprocess).
        2. FFmpeg diminta output berupa RAW VIDEO (piksel mentah,
           format BGR24) ke stdout, BUKAN file/re-encode.
        3. Python baca stdout itu sebagai bytes, lalu diubah jadi
           array numpy yang bentuknya sama persis kayak frame
           yang biasa didapat dari cv2.VideoCapture().read()

    Supaya frame yang dihasilkan tetap bisa dipakai apa adanya
    oleh detector.detect(frame), ocr_reader.read(crop), cv2.imshow(),
    dll -- tanpa perlu ubah kode yang lain.
    """

    def __init__(self, rtmp_url, width, height, ffmpeg_path="ffmpeg"):
        """
        rtmp_url    : URL stream RTMP
        width       : lebar frame asli (dari ffprobe, misal 2688)
        height      : tinggi frame asli (dari ffprobe, misal 1520)
        ffmpeg_path : path ke ffmpeg.exe, default asumsi sudah di PATH
        """

        self.width = width
        self.height = height
        self.rtmp_url = rtmp_url
        self.ffmpeg_path = ffmpeg_path
w
        # Ukuran 1 frame mentah dalam bytes:
        # width * height * 3 channel warna (BGR), 1 byte per channel
        self.frame_size_bytes = width * height * 3

        perintah = [
            ffmpeg_path,

            "-loglevel", "error",   # supaya stdout FFmpeg bersih,
                                     # cuma isi data video (bukan log)

            # Jangan menunggu selamanya jika kamera berhenti mengirim data.
            "-rw_timeout", "15000000",

            "-i", rtmp_url,

            "-an",                  # tidak perlu audio, buang saja

            # PENTING: paksa ukuran output PERSIS width x height yang kita
            # minta. Ini menghindari mismatch ukuran akibat quirk internal
            # H.265 (conformance window / crop yang tidak selalu sama
            # persis dengan metadata width x height dari RTMP). Kalau
            # ukuran tidak dipaksa di sini, hasil reshape() bisa geser
            # sedikit demi sedikit tiap baris -> gambar jadi noise
            # diagonal seperti yang terjadi.
            "-vf", f"scale={width}:{height}",

            "-f", "rawvideo",       # output format: piksel mentah
            "-pix_fmt", "bgr24",    # urutan warna sama seperti OpenCV
            "-",                    # tulis hasil ke stdout
        ]

        self.process = subprocess.Popen(
            perintah,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

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

            sisa_bytes = jumlah_bytes - len(potongan_data)

            chunk = self.process.stdout.read(sisa_bytes)

            if not chunk:

                # FFmpeg berhenti mengirim data -> stream putus
                return None

            potongan_data.extend(chunk)

        return bytes(potongan_data)

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