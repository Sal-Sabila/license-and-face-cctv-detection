-- ============================================================
-- CCTV MONITORING - DATABASE SCHEMA
-- ============================================================
-- Versi revisi dari baseline tim, dengan tambahan:
--   1. ENGINE=InnoDB eksplisit di semua tabel (wajib untuk FK)
--   2. Index tambahan untuk query yang sering dipakai dashboard
--   3. DATETIME(3) untuk presisi milidetik (hindari bentrok
--      waktu saat banyak kendaraan lewat dalam detik yang sama)
--
-- Ada 2 catatan desain yang masih perlu didiskusikan tim,
-- ditandai [PERLU DIPUTUSKAN] di bagian bawah.
-- ============================================================

CREATE DATABASE IF NOT EXISTS cctv_monitoring
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE cctv_monitoring;


-- ============================================================
-- TABEL: cameras
-- ============================================================

CREATE TABLE cameras (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(150),
    stream_url TEXT,
    stream_type ENUM('rtmp','rtsp','http','webcam') DEFAULT 'rtmp',
    status ENUM('online','offline','maintenance') DEFAULT 'offline',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;


-- ============================================================
-- TABEL: detection_events
-- ============================================================

CREATE TABLE detection_events (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    camera_id BIGINT UNSIGNED NOT NULL,
    tracker_id INT NULL,
    object_type ENUM('person','plate') NOT NULL,
    direction ENUM('entry','exit','unknown') DEFAULT 'unknown',
    detection_status ENUM('success','partial','failed') DEFAULT 'success',
    detection_confidence DECIMAL(6,5),
    captured_image_path VARCHAR(500),
    detected_at DATETIME(3) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE,

    -- Query "kejadian per kamera dalam rentang waktu" (dashboard per lokasi)
    INDEX idx_camera_time (camera_id, detected_at),

    -- Query "semua kejadian hari ini / minggu ini" (dashboard umum)
    INDEX idx_detected_at (detected_at)

) ENGINE=InnoDB;


-- ============================================================
-- TABEL: plate_detections
-- ============================================================

CREATE TABLE plate_detections (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    event_id BIGINT UNSIGNED NOT NULL,
    plate_number VARCHAR(30),
    detection_status ENUM('success','failed') NOT NULL,
    detection_confidence DECIMAL(6,5),
    ocr_confidence DECIMAL(6,5),
    raw_ocr_text VARCHAR(100),
    plate_image_path VARCHAR(500),
    detected_at DATETIME(3) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (event_id) REFERENCES detection_events(id) ON DELETE CASCADE,

    -- Fitur pencarian nomor plat -- ini query paling umum di web nanti
    INDEX idx_plate_number (plate_number)

) ENGINE=InnoDB;


-- ============================================================
-- TABEL: face_detections
-- ============================================================

CREATE TABLE face_detections (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    event_id BIGINT UNSIGNED NOT NULL,
    detection_status ENUM('detected','failed') NOT NULL,
    detection_confidence DECIMAL(6,5),
    face_image_path VARCHAR(500),
    detected_at DATETIME(3) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (event_id) REFERENCES detection_events(id) ON DELETE CASCADE

) ENGINE=InnoDB;


-- ============================================================
-- TABEL: detection_logs
-- ============================================================

CREATE TABLE detection_logs (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    event_id BIGINT UNSIGNED NULL,
    camera_id BIGINT UNSIGNED NULL,
    log_type ENUM('person','plate','ocr','face','tracking','system') NOT NULL,
    status ENUM('success','failed','warning','info') NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (event_id) REFERENCES detection_events(id) ON DELETE SET NULL,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL,

    -- Query log per kamera/waktu buat debugging (mirip journalctl tapi di DB)
    INDEX idx_log_camera_time (camera_id, created_at)

) ENGINE=InnoDB;


-- ============================================================
-- TABEL: suspicious_plates
-- ============================================================

CREATE TABLE suspicious_plates (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(30) NOT NULL,
    camera_id BIGINT UNSIGNED NULL,
    first_detected_at DATETIME(3) NOT NULL,
    last_detected_at DATETIME(3) NOT NULL,
    entry_count INT NOT NULL DEFAULT 0,
    exit_count INT NOT NULL DEFAULT 0,
    total_activity INT NOT NULL DEFAULT 0,
    suspicious_reason VARCHAR(255),
    severity ENUM('low','medium','high') DEFAULT 'medium',
    status ENUM('active','resolved','ignored') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE SET NULL,

    -- Lookup cepat plat mencurigakan dari fitur pencarian/alert
    INDEX idx_suspicious_plate (plate_number)

) ENGINE=InnoDB;


-- ============================================================
-- [PERLU DIPUTUSKAN #1] suspicious_plates.camera_id
-- ============================================================
-- Tabel ini nyimpen AGREGAT per plat (entry_count, exit_count,
-- total_activity), tapi camera_id cuma nunjuk ke 1 kamera.
--
-- Pertanyaan buat tim: kalau 1 plat yang sama kedeteksi di
-- BEBERAPA kamera berbeda, gimana cara nyimpennya?
--
-- Opsi A: camera_id di sini maksudnya "kamera pertama kali
--         terdeteksi mencurigakan" -> sebaiknya rename jadi
--         first_camera_id biar jelas
--
-- Opsi B: butuh tracking lintas-kamera -> perlu tabel pivot
--         terpisah, misal:
--
--   CREATE TABLE suspicious_plate_cameras (
--       id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
--       suspicious_plate_id BIGINT UNSIGNED NOT NULL,
--       camera_id BIGINT UNSIGNED NOT NULL,
--       activity_count INT NOT NULL DEFAULT 0,
--       FOREIGN KEY (suspicious_plate_id)
--           REFERENCES suspicious_plates(id) ON DELETE CASCADE,
--       FOREIGN KEY (camera_id)
--           REFERENCES cameras(id) ON DELETE CASCADE
--   ) ENGINE=InnoDB;
--
-- Tergantung definisi "mencurigakan" yang dipakai tim (misal:
-- keluar-masuk gerbang yang sama berkali-kali? atau muncul di
-- banyak lokasi berbeda dalam waktu singkat?)
-- ============================================================


-- ============================================================
-- [PERLU DIPUTUSKAN #2] object_type vs face_detections
-- ============================================================
-- detection_events.object_type cuma punya ENUM('person','plate'),
-- tapi tabel detailnya namanya face_detections (bukan
-- person_detections).
--
-- Kalau yang dimaksud 'person' itu deteksi WAJAH -> sebaiknya
-- ganti jadi ENUM('face','plate') biar konsisten sama nama tabel
--
-- Kalau yang dimaksud 'person' itu deteksi ORANG SECARA UTUH
-- (bukan cuma wajah, misal buat hitung jumlah pejalan kaki) ->
-- mungkin butuh ENUM 3 pilihan: ENUM('person','face','plate'),
-- dengan face_detections khusus buat kasus wajah zoom-in
-- ============================================================