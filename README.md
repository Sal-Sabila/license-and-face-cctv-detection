# Sistem Deteksi AI CCTV

Proyek deteksi cerdas berbasis feed kamera CCTV (RTMP/HLS) dengan dukungan codec H.264 dan H.265/HEVC.

## Struktur Project

```
D:\Bimaa\Magang\CCTV\
│
├── person_detection.py      # Deteksi & pelacakan orang (4 Kamera Live - Grid 2x2)
├── face_detection.py        # Deteksi, embedding, & pengenalan wajah (InsightFace)
├── ffmpeg_reader.py         # Modul universal pembaca stream video (RTMP/HLS/RTSP)
├── test_stream.py           # Script pengujian koneksi stream kamera
│
├── yolov8n.pt               # Bobot model YOLOv8n
├── faces_training.db        # Database embedding wajah
│
├── captures/                # Folder output foto hasil capture orang
├── face_thumbnails/         # Folder output foto wajah terdaftar
├── captured_10_orang/       # Folder dataset foto sampel wajah
└── samples/                 # Folder video sampel offline (deteksi.mp4)
```

## Daftar Kamera Terhubung

Sistem dikonfigurasi untuk memantau 4 kamera CCTV berikut:
1. **CAM 1**: `rtmp://103.255.15.138:1935/live/GSKeluarViewDalam.stream`
2. **CAM 2**: `rtmp://103.255.15.138:1935/live/GSKeluarViewLuar.stream`
3. **CAM 3**: `rtmp://103.255.15.138:1935/live/GSMasukViewDalam.stream`
4. **CAM 4**: `rtmp://103.255.15.138:1935/live/GSMasukViewLuar.stream`

---

## Cara Penggunaan

### 1. Menjalankan Deteksi Orang (4 CCTV - Grid 2x2)
Membuka 4 kamera CCTV live sekaligus dalam tata letak 2x2, mendeteksi orang di setiap kamera, melacak pergerakan, dan otomatis menyimpan foto crop orang baru ke folder `captures/`:
```bash
python person_detection.py
```
*Tampilan:*
- **Grid 2x2** (resolusi 1280x758)
- **Status Bar Atas**: Total orang di seluruh frame, total capture seluruh kamera, FPS gabungan.
- **Header Setiap Tile**: Nama kamera, indikator LIVE hijau, jumlah orang di frame kamera tersebut, dan jumlah capture kamera tersebut.
- Tekan **`q`** untuk berhenti dan melihat ringkasan total orang yang ditangkap per kamera.

### 2. Menjalankan Deteksi Wajah (Face Detection)
Mendeteksi wajah, mencocokkan kemiripan wajah dengan database, dan mendaftarkan wajah baru ke `faces_training.db`:
```bash
python face_detection.py
```

### 3. Uji Coba Koneksi Stream
Untuk mengetes feed kamera secara individual:
```bash
python test_stream.py
```
