import time
import cv2
import csv
import io
import numpy as np
from flask import Blueprint, jsonify, request, Response
from datetime import datetime
import db
from ffmpeg_stream_reader import FFmpegStreamReader, normalize_stream_url

plate_bp = Blueprint("plate", __name__)


def parse_confidence(data):
    try:
        return float(data.get("confidence"))
    except (TypeError, ValueError):
        return None


# ============================================================
# ENDPOINT KAMERA (TERHUBUNG KE MYSQL REAL_CCTV)
# ============================================================

@plate_bp.route("/cameras", methods=["GET"])
def list_cameras():
    """Mengambil daftar kamera langsung dari MySQL real_cctv."""
    try:
        cameras_db = db.get_all_cameras()
        formatted = [
            {
                "id": c["camera_id"],
                "name": c["location"],
                "url": c["stream_url"],
                "stream_type": c.get("stream_type", 1),
                "active": bool(c["status"])
            }
            for c in cameras_db
        ]
        return jsonify({"success": True, "data": formatted})
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "data": []}), 500


@plate_bp.route("/cameras", methods=["POST"])
def create_camera():
    """Menambah kamera baru ke database MySQL."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    url = str(data.get("url", "")).strip()
    active = bool(data.get("active", True))

    if not name or not url:
        return jsonify({"success": False, "message": "Nama dan URL kamera wajib diisi"}), 400

    try:
        new_id = db.add_camera(location=name, stream_url=url, stream_type=1, status=1 if active else 0)
        return jsonify({
            "success": True,
            "data": {
                "id": new_id,
                "name": name,
                "url": url,
                "active": active
            }
        }), 201
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/cameras/<int:camera_id>", methods=["PUT"])
def update_camera(camera_id):
    """Memperbarui informasi kamera di database MySQL."""
    data = request.get_json(silent=True) or {}
    location = data.get("name")
    stream_url = data.get("url")
    status = 1 if data.get("active") else (0 if "active" in data else None)

    try:
        ok = db.update_camera(camera_id, location=location, stream_url=stream_url, status=status)
        if not ok:
            return jsonify({"success": False, "message": "Kamera tidak ditemukan atau tidak ada perubahan"}), 404
        return jsonify({"success": True, "message": "Kamera berhasil diperbarui"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/cameras/<int:camera_id>", methods=["DELETE"])
def delete_camera(camera_id):
    """Menghapus kamera dari database MySQL."""
    try:
        ok = db.delete_camera(camera_id)
        if not ok:
            return jsonify({"success": False, "message": "Kamera tidak ditemukan"}), 404
        return jsonify({"success": True, "message": "Kamera berhasil dihapus"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# ENDPOINT HASIL DETEKSI GABUNGAN (TERHUBUNG KE MYSQL REAL_CCTV)
# ============================================================

@plate_bp.route("/detections", methods=["GET"])
def list_detections():
    """Mengambil daftar deteksi gabungan (Plat & Wajah) dengan paginasi dan filter."""
    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 20, type=int)
    type_filter = request.args.get("type", "all")
    status_filter = request.args.get("status", "all")
    camera_id = request.args.get("camera_id", type=int)
    search = request.args.get("search", type=str)
    start_date = request.args.get("start_date", type=str)
    end_date = request.args.get("end_date", type=str)

    try:
        res = db.get_all_detections_paginated(
            page=page,
            limit=limit,
            type_filter=type_filter,
            status_filter=status_filter,
            camera_id=camera_id,
            search=search,
            start_date=start_date,
            end_date=end_date
        )
        return jsonify({
            "success": True,
            "data": res["items"],
            "total": res["total"],
            "page": res["page"],
            "limit": res["limit"],
            "total_pages": res["total_pages"]
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "data": []}), 500


@plate_bp.route("/detections/<int:detection_id>", methods=["DELETE"])
def delete_detection(detection_id):
    """Menghapus satu histori deteksi beserta data turunannya."""
    try:
        if not db.delete_detection(detection_id):
            return jsonify({"success": False, "message": "Deteksi tidak ditemukan"}), 404
        return jsonify({"success": True, "message": "Deteksi berhasil dihapus"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# ENDPOINT RIWAYAT DETEKSI PLAT & WAJAH
# ============================================================

@plate_bp.route("/plate/history", methods=["GET"])
def plate_history():
    """Mengambil riwayat deteksi plat nomor dari database MySQL dengan filter & paginasi."""
    # Dukung paginasi jika parameter page atau search dikirimkan
    if "page" in request.args or "search" in request.args or "limit" in request.args:
        page = request.args.get("page", 1, type=int)
        limit = request.args.get("limit", 20, type=int)
        search = request.args.get("search", type=str)
        camera_id = request.args.get("camera_id", type=int)
        status_filter = request.args.get("status", "all")
        start_date = request.args.get("start_date", type=str)
        end_date = request.args.get("end_date", type=str)

        try:
            res = db.get_plate_history_paginated(
                page=page,
                limit=limit,
                search=search,
                camera_id=camera_id,
                status_filter=status_filter,
                start_date=start_date,
                end_date=end_date
            )
            return jsonify({
                "success": True,
                "data": res["items"],
                "total": res["total"],
                "page": res["page"],
                "limit": res["limit"],
                "total_pages": res["total_pages"]
            })
        except Exception as e:
            return jsonify({"success": False, "message": str(e), "data": []}), 500

    # Default legacy endpoint (mengambil list 100 terbaru)
    try:
        records = db.get_recent_detections(limit=100)
        plate_list = []
        for r in records:
            if r.get("plate_number"):
                status_text = "Terbaca" if r.get("plate_status") == 1 else ("Perlu cek" if r.get("plate_status") == 2 else "Gagal")
                conf = float(r.get("plate_confidence") or 0.0)
                dt_str = r["detected_at"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(r.get("detected_at"), datetime) else str(r.get("detected_at") or "")

                plate_list.append({
                    "id": r["detection_id"],
                    "plate": r["plate_number"],
                    "camera": r.get("camera_name") or "CCTV",
                    "confidence": conf,
                    "confidence_percent": round(conf * 100, 1),
                    "timestamp": dt_str,
                    "status": status_text,
                    "image_path": r.get("plate_image_path")
                })
        return jsonify({"success": True, "data": plate_list})
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "data": []}), 500


@plate_bp.route("/plate/history/<int:plate_id>", methods=["DELETE"])
def delete_plate_history(plate_id):
    """Menghapus satu histori plat beserta event dan capture terkait."""
    try:
        if not db.delete_plate(plate_id):
            return jsonify({"success": False, "message": "Riwayat plat tidak ditemukan"}), 404
        return jsonify({"success": True, "message": "Riwayat plat berhasil dihapus"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/plate/latest", methods=["GET"])
def latest_plate():
    """Mengambil deteksi plat paling akhir."""
    res = plate_history().get_json()
    items = res.get("data", [])
    return jsonify({"success": True, "data": items[0] if items else None})


@plate_bp.route("/face/history", methods=["GET"])
def face_history():
    """Mengambil riwayat deteksi wajah/orang dari database MySQL."""
    try:
        records = db.get_recent_detections(limit=100)
        face_list = []
        for r in records:
            if r.get("face_image_path") or r.get("face_status") is not None:
                status_text = "Terbaca" if r.get("face_status") == 1 else ("Perlu cek" if r.get("face_status") == 2 else "Gagal")
                conf = float(r.get("face_confidence") or 0.0)
                dt_str = r["detected_at"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(r.get("detected_at"), datetime) else str(r.get("detected_at") or "")

                face_list.append({
                    "id": r["detection_id"],
                    "face_count": 1,
                    "camera": r.get("camera_name") or "CCTV",
                    "confidence": conf,
                    "confidence_percent": round(conf * 100, 1),
                    "timestamp": dt_str,
                    "status": status_text,
                    "image_path": r.get("face_image_path")
                })
        return jsonify({"success": True, "data": face_list})
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "data": []}), 500


@plate_bp.route("/face/latest", methods=["GET"])
def latest_face():
    """Mengambil deteksi wajah paling akhir."""
    res = face_history().get_json()
    items = res.get("data", [])
    return jsonify({"success": True, "data": items[0] if items else None})


# ============================================================
# ENDPOINT STATISTIK DASHBOARD & ENTERPRISE
# ============================================================

@plate_bp.route("/statistics/summary", methods=["GET"])
def stats_summary():
    """Statistik ringkasan agregat langsung dari MySQL."""
    try:
        stats = db.get_dashboard_stats()
        return jsonify({"success": True, "data": stats})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/statistics/enterprise", methods=["GET"])
def stats_enterprise():
    """
    Statistik analitik komprehensif standar perusahaan:
    - 6 KPI Korporat
    - Time-series trend (per jam atau per hari)
    - Distribusi beban CCTV
    - Status breakdown SLA (Donut)
    - Analisis Jam Sibuk
    - Top 10 Plat Kendaraan
    """
    period = request.args.get("period", "today")
    try:
        data = db.get_enterprise_statistics(period=period)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# ENDPOINT PENGATURAN SISTEM (TERHUBUNG KE MYSQL REAL_CCTV)
# ============================================================

@plate_bp.route("/settings", methods=["GET"])
def get_settings():
    """Mengambil konfigurasi sistem & status diagnostik dari MySQL."""
    try:
        settings = db.get_system_settings()
        diagnostics = db.get_system_diagnostics()
        return jsonify({
            "success": True,
            "settings": settings,
            "diagnostics": diagnostics
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/settings", methods=["POST"])
def save_settings():
    """Menyimpan konfigurasi sistem ke tabel system_settings di database MySQL."""
    data = request.get_json(silent=True) or {}
    try:
        db.update_system_settings(data)
        return jsonify({
            "success": True,
            "message": "Pengaturan sistem berhasil disimpan ke database MySQL."
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/settings/seed-demo", methods=["POST"])
def seed_demo():
    """Mengisi database dengan data simulasi realistis untuk keperluan pengujian/presentasi."""
    data = request.get_json(silent=True) or {}
    count = int(data.get("count", 35))
    try:
        inserted = db.seed_demo_data(count=count)
        return jsonify({
            "success": True,
            "message": f"{inserted} data deteksi simulasi berhasil ditambahkan ke database.",
            "inserted": inserted
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# ENDPOINT EKSPOR DATA KE CSV (STANDAR LAPORAN AUDIT PERUSAHAAN)
# ============================================================

@plate_bp.route("/export/detections", methods=["GET"])
def export_detections_csv():
    """Mengekspor seluruh data deteksi ke format CSV."""
    try:
        res = db.get_all_detections_paginated(page=1, limit=5000)
        items = res.get("items", [])

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Tipe", "Nomor Plat", "Kamera CCTV", "Confidence (%)", "Waktu Deteksi", "Status"])

        for it in items:
            writer.writerow([
                it["id"],
                it["type"],
                it["plate"],
                it["camera"],
                it["confidence_percent"],
                it["timestamp"],
                it["status"]
            ])

        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=laporan_deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/plates", methods=["GET"])
def export_plates_csv():
    """Mengekspor riwayat plat nomor ke format CSV."""
    try:
        res = db.get_plate_history_paginated(page=1, limit=5000)
        items = res.get("items", [])

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID Plat", "Nomor Plat", "Kamera CCTV", "Confidence (%)", "Waktu Deteksi", "Status"])

        for it in items:
            writer.writerow([
                it["id"],
                it["plate"],
                it["camera"],
                it["confidence_percent"],
                it["timestamp"],
                it["status"]
            ])

        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=riwayat_plat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/statistics", methods=["GET"])
def export_statistics_csv():
    """Mengekspor laporan statistik analitik eksekutif ke CSV."""
    period = request.args.get("period", "today")
    try:
        data = db.get_enterprise_statistics(period=period)
        kpi = data.get("kpi", {})
        top_plates = data.get("top_plates", [])
        cam_dist = data.get("camera_distribution", [])

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["LAPORAN EKSEKUTIF ANALITIK CCTV - PLATEVISION"])
        writer.writerow(["Periode", data.get("period_label", period)])
        writer.writerow(["Tanggal Cetak", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])

        writer.writerow(["RINGKASAN KPI UTAMA"])
        writer.writerow(["Metrik", "Nilai"])
        writer.writerow(["Total Deteksi", kpi.get("total_detections", 0)])
        writer.writerow(["Total Plat Nomor", kpi.get("total_plates", 0)])
        writer.writerow(["Total Wajah / Orang", kpi.get("total_faces", 0)])
        writer.writerow(["Plate Recognition Rate (%)", f"{kpi.get('plate_read_rate', 0)}%"])
        writer.writerow(["Rata-rata Akurasi AI (%)", f"{kpi.get('avg_confidence', 0)}%"])
        writer.writerow(["Perlu Review Manual", kpi.get("need_check_count", 0)])
        writer.writerow(["Kamera Aktif", f"{kpi.get('active_cameras', 0)} / {kpi.get('total_cameras', 0)}"])
        writer.writerow([])

        writer.writerow(["DISTRIBUSI LALU LINTAS PER KAMERA CCTV"])
        writer.writerow(["Nama CCTV", "Total Deteksi", "Plat", "Wajah", "Pangsa (%)"])
        for c in cam_dist:
            writer.writerow([c["camera_name"], c["total_count"], c["plate_count"], c["face_count"], f"{c['percentage']}%"])
        writer.writerow([])

        writer.writerow(["TOP 10 PLAT PALING SERING TERDETEKSI"])
        writer.writerow(["Nomor Plat", "Frekuensi", "Lokasi Terakhir", "Waktu Terakhir", "Status"])
        for tp in top_plates:
            writer.writerow([tp["plate_number"], tp["total_seen"], tp["last_camera"], tp["last_seen"], tp["status"]])

        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=laporan_statistik_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# ENDPOINT LIVE CCTV STREAM (MJPEG UNTUK BROWSER)
# ============================================================

def generate_mjpeg_stream(stream_url, width=960, height=540, draw_bbox=True, camera_id=1):
    """Membaca frame dari CCTV via FFmpegStreamReader dan stream MJPEG ke browser dengan AI bounding box."""
    norm_url = normalize_stream_url(stream_url)
    reader = FFmpegStreamReader(norm_url, width=width, height=height)

    ai_service = None
    try:
        from services.stream_ai_service import StreamAIService
        # AI tetap dijalankan walaupun visual bounding box dimatikan.
        ai_service = StreamAIService.get_instance()
    except Exception as e:
        print(f"[AI STREAM WARNING] AI Service load error: {e}")

    try:
        failed_reads = 0
        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads < 3 and reader.isOpened():
                    time.sleep(0.2)
                    continue

                error_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(error_frame, "STREAM CCTV TIDAK TERSEDIA", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 180, 255), 2)
                error_message = reader.error_message or "FFmpeg tidak menerima frame dari sumber CCTV"
                cv2.putText(error_frame, error_message[:110], (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                break

            # Jalankan deteksi & gambar bounding box jika aktif
            if ai_service is not None:
                try:
                    frame = ai_service.process_frame(frame, draw_bbox=draw_bbox, camera_id=camera_id)
                except Exception as e:
                    print(f"[AI STREAM ERROR] Frame processing failed: {e}")

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)  # cap ~30 FPS untuk kestabilan CPU
    except GeneratorExit:
        pass
    except Exception:
        pass
    finally:
        reader.release()


@plate_bp.route("/video_feed")
@plate_bp.route("/video_feed/<int:camera_id>")
def video_feed(camera_id=None):
    """Endpoint feed video real-time untuk elemen <img class='live-stream-feed'> di browser."""
    draw_bbox = request.args.get("bbox", "1").lower() in ("1", "true", "yes", "on")

    cams = db.get_all_cameras()
    target_cam = None

    if camera_id is not None:
        target_cam = next((c for c in cams if c["camera_id"] == camera_id), None)

    # Jika tidak ditentukan atau tidak ketemu, pakai kamera aktif pertama
    if not target_cam:
        target_cam = next((c for c in cams if c["status"] == 1), None)

    # Fallback ke kamera pertama jika ada
    if not target_cam and cams:
        target_cam = cams[0]

    if not target_cam or not target_cam.get("stream_url"):
        return "Kamera tidak ditemukan atau belum aktif", 404

    return Response(
        generate_mjpeg_stream(
            target_cam["stream_url"],
            draw_bbox=draw_bbox,
            camera_id=target_cam["camera_id"]
        ),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

