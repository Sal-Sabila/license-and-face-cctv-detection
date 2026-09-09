import time
import cv2
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
# ENDPOINT RIWAYAT DETEKSI (TERHUBUNG KE MYSQL REAL_CCTV)
# ============================================================

@plate_bp.route("/plate/history", methods=["GET"])
def plate_history():
    """Mengambil riwayat deteksi plat nomor dari database MySQL."""
    try:
        records = db.get_recent_detections(limit=100)
        plate_list = []
        for r in records:
            if r.get("plate_number"):
                # Status: 1=Terbaca (success), 2=Perlu cek (warning), 0=Gagal
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
# ENDPOINT STATISTIK DASHBOARD
# ============================================================

@plate_bp.route("/statistics/summary", methods=["GET"])
def stats_summary():
    """Statistik agregat langsung dari MySQL."""
    try:
        stats = db.get_dashboard_stats()
        return jsonify({"success": True, "data": stats})
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
    if draw_bbox:
        try:
            from services.stream_ai_service import StreamAIService
            ai_service = StreamAIService.get_instance()
        except Exception as e:
            print(f"[AI STREAM WARNING] AI Service load error: {e}")

    try:
        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                time.sleep(0.04)
                continue

            # Jalankan deteksi & gambar bounding box jika aktif
            if ai_service is not None and draw_bbox:
                try:
                    frame = ai_service.process_frame(frame, draw_bbox=True, camera_id=camera_id)
                except Exception as e:
                    pass

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

