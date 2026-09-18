import time
import cv2
import numpy as np
import os
import threading
import uuid
import queue
from flask import Blueprint, jsonify, request, Response, send_from_directory, send_file
from datetime import datetime
import db
from ffmpeg_stream_reader import FFmpegStreamReader, normalize_stream_url
from report_export import (
    build_detections_excel,
    build_detections_pdf,
    build_plates_excel,
    build_plates_pdf,
    build_recap_excel,
    build_recap_pdf,
    build_statistics_excel,
    build_statistics_pdf,
    build_cameras_excel,
    build_cameras_pdf,
)

plate_bp = Blueprint("plate", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
VIDEO_UPLOAD_DIR = os.path.join(VIDEO_DIR, "uploads")
VIDEO_OUTPUT_DIR = os.path.join(VIDEO_DIR, "processed")
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
VIDEO_JOBS = {}
VIDEO_JOBS_LOCK = threading.Lock()

os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


def _video_job(job_id, input_path, output_path, camera_id):
    try:
        from services.video_ai_service import VideoAIService

        def update_progress(frame_number, total_frames, processed_frame=None):
            progress = round((frame_number / total_frames) * 100, 1) if total_frames else 0
            with VIDEO_JOBS_LOCK:
                if job_id in VIDEO_JOBS:
                    VIDEO_JOBS[job_id]["progress"] = progress
                    if processed_frame is not None:
                        encoded, buffer = cv2.imencode(
                            ".jpg", processed_frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 78]
                        )
                        if encoded:
                            VIDEO_JOBS[job_id]["latest_frame"] = buffer.tobytes()

        with VIDEO_JOBS_LOCK:
            VIDEO_JOBS[job_id]["status"] = "processing"

        service = VideoAIService(
            video_path=input_path,
            output_path=output_path,
            camera_id=camera_id,
            show_window=False,
            progress_callback=update_progress
        )
        service.run()

        with VIDEO_JOBS_LOCK:
            VIDEO_JOBS[job_id].update({
                "status": "completed",
                "progress": 100,
                "output_url": f"/api/video_jobs/{job_id}/result"
            })
    except Exception as exc:
        print(f"[VIDEO JOB ERROR] {job_id}: {exc}")
        with VIDEO_JOBS_LOCK:
            VIDEO_JOBS[job_id].update({
                "status": "failed",
                "error": str(exc)
            })
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


def parse_confidence(data):
    try:
        return float(data.get("confidence"))
    except (TypeError, ValueError):
        return None


@plate_bp.route("/video_jobs", methods=["POST"])
def create_video_job():
    """Menerima video dan memprosesnya di background agar request web tidak tertahan."""
    uploaded = request.files.get("video")
    if not uploaded or not uploaded.filename:
        return jsonify({"success": False, "message": "File video wajib dipilih"}), 400

    extension = uploaded.filename.rsplit(".", 1)[-1].lower() if "." in uploaded.filename else ""
    if extension not in VIDEO_EXTENSIONS:
        return jsonify({"success": False, "message": "Format video tidak didukung"}), 400

    try:
        camera_id = int(request.form.get("camera_id", 1))
    except (TypeError, ValueError):
        camera_id = 1

    job_id = uuid.uuid4().hex
    input_path = os.path.join(VIDEO_UPLOAD_DIR, f"{job_id}.{extension}")
    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{job_id}.mp4")
    uploaded.save(input_path)

    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "output_url": None,
            "error": None,
            "latest_frame": None
        }

    worker = threading.Thread(
        target=_video_job,
        args=(job_id, input_path, output_path, camera_id),
        daemon=True
    )
    worker.start()

    return jsonify({"success": True, "job_id": job_id}), 202


@plate_bp.route("/video_jobs/<job_id>", methods=["GET"])
def video_job_status(job_id):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if not job:
            return jsonify({"success": False, "message": "Job video tidak ditemukan"}), 404
        status = {
            key: value
            for key, value in job.items()
            if key != "latest_frame"
        }
        return jsonify({"success": True, "data": status})


@plate_bp.route("/video_jobs/<job_id>/result", methods=["GET"])
def video_job_result(job_id):
    filename = f"{job_id}.mp4"
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if not job or job.get("status") != "completed":
            return jsonify({"success": False, "message": "Video belum selesai diproses"}), 404
    return send_from_directory(VIDEO_OUTPUT_DIR, filename, as_attachment=False)


@plate_bp.route("/video_jobs/<job_id>/feed", methods=["GET"])
def video_job_feed(job_id):
    """Feed MJPEG frame deteksi terbaru selama video masih diproses."""
    def generate_frames():
        last_frame = None
        while True:
            with VIDEO_JOBS_LOCK:
                job = VIDEO_JOBS.get(job_id)
                if not job:
                    return
                status = job["status"]
                frame = job.get("latest_frame")

            if frame and frame != last_frame:
                last_frame = frame
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            if status in ("completed", "failed"):
                return
            time.sleep(0.12)

    with VIDEO_JOBS_LOCK:
        if job_id not in VIDEO_JOBS:
            return jsonify({"success": False, "message": "Job video tidak ditemukan"}), 404

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


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
                "active": bool(c["status"]),
                "direction": c.get("direction") or "unknown"
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
    direction = str(data.get("direction", "unknown"))

    if not name or not url:
        return jsonify({"success": False, "message": "Nama dan URL kamera wajib diisi"}), 400

    try:
        new_id = db.add_camera(location=name, stream_url=url, stream_type=1, status=1 if active else 0, direction=direction)
        return jsonify({
            "success": True,
            "data": {
                "id": new_id,
                "name": name,
                "url": url,
                "active": active,
                "direction": direction
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
    direction = data.get("direction")

    try:
        ok = db.update_camera(camera_id, location=location, stream_url=stream_url, status=status, direction=direction)
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
            if r.get("object_type") == "person" or (not r.get("object_type") and r.get("face_image_path") and not r.get("plate_id")):
                status_text = "Terbaca" if r.get("face_status") == 1 else ("Perlu cek" if r.get("face_status") == 2 else "Gagal")
                conf = float(r.get("person_confidence") or r.get("face_confidence") or 0.0)
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


@plate_bp.route("/analytics", methods=["GET"])
def analytics():
    """Shared filtered analytics payload for dashboard, recap and statistics."""
    try:
        return jsonify({"success": True, "data": db.get_analytics(request.args.to_dict())})
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "data": {}}), 500


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
        analytics = db.get_analytics(request.args.to_dict())
        summary = analytics["summary"]
        daily = analytics["daily"]
        hourly = analytics["hourly"]
        return jsonify({"success": True, "data": {
            "period": analytics["period"],
            "period_label": period,
            "kpi": {
                "total_detections": summary["vehicles"] + summary["people"],
                "total_plates": summary["plates"],
                "total_faces": summary["people"],
                "plate_read_rate": round(summary["plates"] / max(summary["vehicles"], 1) * 100, 1),
                "avg_confidence": 0,
                "need_check_count": 0,
                "active_cameras": summary["active_cameras"],
                "total_cameras": summary["total_cameras"],
                "camera_availability": round(summary["active_cameras"] / max(summary["total_cameras"], 1) * 100, 1)
            },
            "trend": {"labels": [item["date"] for item in daily], "plates": [item["vehicles"] for item in daily], "faces": [item["people"] for item in daily], "totals": [item["vehicles"] + item["people"] for item in daily]},
            "camera_distribution": [{"camera_name": item["camera"], "total_count": item["vehicles"] + item["people"], "plate_count": item["unique_plates"], "face_count": item["people"], "percentage": 0} for item in analytics["cameras"]],
            "status_breakdown": {"valid": 0, "warning": 0, "failed": 0},
            "peak_hours": [{"time_range": f"{item['label']} - {(item['hour'] + 1) % 24:02d}:00 WIB", "count": item["vehicles"], "percentage": 0} for item in sorted(hourly, key=lambda value: value["vehicles"], reverse=True)[:3]],
            "top_plates": [{"plate_number": item["plate"], "total_seen": item["count"], "last_camera": item["camera"], "last_seen": item["last_seen"], "status": "Aktual", "status_code": 1, "avg_confidence_percent": item["confidence"]} for item in analytics["top_plates"]]
        }})
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
# ENDPOINT EKSPOR DATA KE EXCEL (.xlsx) & PDF
# Excel dibuat dengan openpyxl, PDF dibuat dengan ReportLab,
# keduanya lewat report_export.py agar konsisten dengan data
# yang memang ditampilkan di halaman masing-masing.
# ============================================================

EXCEL_MIMETYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@plate_bp.route("/export/detections", methods=["GET"])
def export_detections_excel():
    """Mengekspor seluruh data Hasil Deteksi ke file Excel (.xlsx) menggunakan openpyxl."""
    try:
        res = db.get_all_detections_paginated(
            page=1,
            limit=5000,
            start_date=request.args.get("start_date", type=str),
            end_date=request.args.get("end_date", type=str)
        )
        items = res.get("items", [])
        buffer = build_detections_excel(items)
        filename = f"laporan_deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            buffer,
            mimetype=EXCEL_MIMETYPE,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/detections/pdf", methods=["GET"])
def export_detections_pdf():
    """Mengekspor seluruh data Hasil Deteksi ke file PDF (dengan foto) menggunakan ReportLab."""
    try:
        res = db.get_all_detections_paginated(
            page=1,
            limit=5000,
            start_date=request.args.get("start_date", type=str),
            end_date=request.args.get("end_date", type=str)
        )
        items = res.get("items", [])
        buffer = build_detections_pdf(items)
        filename = f"laporan_deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/plates", methods=["GET"])
def export_plates_excel():
    """Mengekspor riwayat plat nomor ke file Excel (.xlsx) menggunakan openpyxl."""
    try:
        res = db.get_plate_history_paginated(
            page=1,
            limit=5000,
            start_date=request.args.get("start_date", type=str),
            end_date=request.args.get("end_date", type=str)
        )
        items = res.get("items", [])
        buffer = build_plates_excel(items)
        filename = f"riwayat_plat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            buffer,
            mimetype=EXCEL_MIMETYPE,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/plates/pdf", methods=["GET"])
def export_plates_pdf():
    """Mengekspor riwayat plat nomor ke file PDF (dengan crop plat) menggunakan ReportLab."""
    try:
        res = db.get_plate_history_paginated(
            page=1,
            limit=5000,
            start_date=request.args.get("start_date", type=str),
            end_date=request.args.get("end_date", type=str)
        )
        items = res.get("items", [])
        buffer = build_plates_pdf(items)
        filename = f"riwayat_plat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/recap", methods=["GET"])
def export_recap_excel():
    """Mengekspor Rekapitulasi ke Excel (.xlsx) - 1 file, 3 sheet: Ringkasan, Rekap CCTV, Total Harian."""
    try:
        analytics = db.get_analytics(request.args.to_dict())
        buffer = build_recap_excel(analytics)
        filename = f"rekapitulasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            buffer,
            mimetype=EXCEL_MIMETYPE,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/recap/pdf", methods=["GET"])
def export_recap_pdf():
    """Mengekspor Rekapitulasi ke PDF (Ringkasan, Rekap CCTV, Total Harian) menggunakan ReportLab."""
    try:
        analytics = db.get_analytics(request.args.to_dict())
        buffer = build_recap_pdf(analytics)
        filename = f"rekapitulasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/statistics", methods=["GET"])
def export_statistics_excel():
    """Mengekspor Statistik ke Excel (.xlsx) - 1 file, 2 sheet: Statistik, Top 10."""
    period = request.args.get("period", "today")
    try:
        analytics = db.get_analytics(request.args.to_dict())
        buffer = build_statistics_excel(analytics, period=analytics.get("period", period))
        filename = f"laporan_statistik_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            buffer,
            mimetype=EXCEL_MIMETYPE,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/statistics/pdf", methods=["GET"])
def export_statistics_pdf():
    """Mengekspor Statistik ke PDF (Statistik Utama, Beban Lalu Lintas, Jam Sibuk, Top 10) menggunakan ReportLab."""
    period = request.args.get("period", "today")
    try:
        analytics = db.get_analytics(request.args.to_dict())
        buffer = build_statistics_pdf(analytics, period=analytics.get("period", period))
        filename = f"laporan_statistik_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# ENDPOINT EKSPOR MONITORING CCTV (EXCEL .xlsx & PDF)
# Data diambil dari db.get_all_cameras() -- fungsi yang sama
# dipakai endpoint GET /cameras -- agar data website, Excel,
# dan PDF selalu konsisten.
# ============================================================

@plate_bp.route("/export/cameras", methods=["GET"])
def export_cameras_excel():
    """Mengekspor daftar kamera CCTV ke file Excel (.xlsx) menggunakan openpyxl."""
    try:
        cameras_db = db.get_all_cameras()
        buffer = build_cameras_excel(cameras_db)
        filename = f"monitoring_cctv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            buffer,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@plate_bp.route("/export/cameras/pdf", methods=["GET"])
def export_cameras_pdf():
    """Mengekspor daftar kamera CCTV ke file PDF menggunakan ReportLab."""
    try:
        cameras_db = db.get_all_cameras()
        buffer = build_cameras_pdf(cameras_db)
        filename = f"monitoring_cctv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
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

    # Pembacaan dan pengiriman frame tidak boleh menunggu inferensi AI.
    # Queue satu item menjaga latency tetap rendah saat CPU sedang penuh.
    ai_input = queue.Queue(maxsize=1)
    ai_output = queue.Queue(maxsize=1)
    stop_worker = threading.Event()

    def run_ai_worker():
        while not stop_worker.is_set():
            try:
                source_frame = ai_input.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result_frame = ai_service.process_frame(
                    source_frame,
                    draw_bbox=draw_bbox,
                    camera_id=camera_id
                )
                while True:
                    try:
                        ai_output.get_nowait()
                    except queue.Empty:
                        break
                if result_frame is not None and getattr(result_frame, "size", 0):
                    ai_output.put_nowait(result_frame)
                else:
                    print(f"[AI STREAM ERROR] Invalid processed frame for camera={camera_id}")
            except Exception as e:
                print(f"[AI STREAM ERROR] Frame processing failed: {e}")

    worker = None
    if ai_service is not None:
        worker = threading.Thread(target=run_ai_worker, daemon=True)
        worker.start()

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

            # Jalankan AI di worker; frame terbaru tetap dikirim tanpa menunggu.
            if ai_service is not None:
                try:
                    while True:
                        ai_input.get_nowait()
                except queue.Empty:
                    pass
                try:
                    ai_input.put_nowait(frame.copy())
                except Exception as e:
                    print(f"[AI STREAM ERROR] Queue frame failed: {e}")

                try:
                    frame = ai_output.get_nowait()
                except queue.Empty:
                    pass

            if frame is None or not hasattr(frame, "size") or frame.size == 0:
                print(f"[STREAM ERROR] Empty frame skipped for camera={camera_id}")
                continue
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                print(f"[STREAM ERROR] JPEG encode failed for camera={camera_id}")
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)  # cap ~30 FPS untuk kestabilan CPU
    except GeneratorExit:
        pass
    except Exception as exc:
        print(f"[STREAM ERROR] MJPEG generator failed for camera={camera_id}: {exc}")
    finally:
        stop_worker.set()
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