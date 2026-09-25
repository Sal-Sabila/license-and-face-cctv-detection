from flask import Blueprint, request, jsonify
import json
import db

zones_bp = Blueprint("zones_page", __name__)


# ============================================================
# ZONA DEFAULT PER KAMERA
# ============================================================
# ✅ FIX: Zona baru dinaikkan supaya motor/pejalan di tengah frame
# tetap masuk zona NEAR. Sebelumnya y=0.35/0.38 membuat motor di
# y=0.11-0.24 (dari log) dianggap FAR dan di-skip.
# ------------------------------------------------------------
# Sebelumnya:
#   DEFAULT_MID  = [(0.10, 0.35), (0.90, 0.35), (0.99, 1.00), (0.01, 1.00)]
#   DEFAULT_NEAR = [(0.10, 0.38), (0.90, 0.38), (0.99, 1.00), (0.01, 1.00)]
# ------------------------------------------------------------

DEFAULT_MID = [
    (0.05, 0.10),
    (0.95, 0.10),
    (0.99, 1.00),
    (0.01, 1.00),
]

DEFAULT_NEAR = [
    (0.05, 0.20),
    (0.95, 0.20),
    (0.99, 1.00),
    (0.01, 1.00),
]


def _validate_zone(name, poly):
    """Validasi polygon: minimal 3 titik, koordinat 0..1."""
    if not poly:
        return f"{name.upper()} wajib diisi."
    if len(poly) < 3:
        return f"{name.upper()} minimal 3 titik."
    for point in poly:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return f"{name.upper()} titik tidak valid."
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return f"{name.upper()} koordinat bukan angka."
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return f"{name.upper()} koordinat harus 0..1."
    return None


def _check_self_intersect(name, poly):
    """Cek polygon self-intersect (butuh shapely)."""
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return None  # Skip kalau shapely tidak terinstall

    try:
        p = Polygon(poly)
        if not p.is_valid:
            return f"Zona {name.upper()} menyilang (self-intersect). Perbaiki urutan titik."
        if p.area < 0.01:
            return f"Zona {name.upper()} terlalu kecil. Perbesar polygon."
    except Exception:
        return None
    return None


@zones_bp.route("/api/zones/<int:camera_id>", methods=["GET"])
def get_zone(camera_id):
    """Ambil zona MID & NEAR untuk kamera tertentu."""
    try:
        zone = db.get_camera_zone(camera_id)
        if zone is None:
            return jsonify({
                "success": True,
                "data": {
                    "camera_id": camera_id,
                    "mid": DEFAULT_MID,
                    "near": DEFAULT_NEAR,
                    "is_custom": False,
                }
            })
        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "mid": zone["mid"],
                "near": zone["near"],
                "is_custom": True,
            }
        })
    except Exception as exc:
        print(f"[ZONES GET ERROR] {exc}")
        return jsonify({"success": False, "message": str(exc)}), 500


@zones_bp.route("/api/zones/<int:camera_id>", methods=["POST"])
def save_zone(camera_id):
    """Simpan zona MID & NEAR untuk kamera tertentu."""
    try:
        payload = request.get_json(force=True) or {}
        mid = payload.get("mid")
        near = payload.get("near")

        # Validasi dasar
        err = _validate_zone("mid", mid)
        if err:
            print(f"[ZONES] Validasi gagal CAM {camera_id}: {err}")
            return jsonify({"success": False, "message": err}), 400

        err = _validate_zone("near", near)
        if err:
            print(f"[ZONES] Validasi gagal CAM {camera_id}: {err}")
            return jsonify({"success": False, "message": err}), 400

        # Validasi self-intersect & luas (butuh shapely)
        err = _check_self_intersect("mid", mid)
        if err:
            print(f"[ZONES] Validasi gagal CAM {camera_id}: {err}")
            return jsonify({"success": False, "message": err}), 400

        err = _check_self_intersect("near", near)
        if err:
            print(f"[ZONES] Validasi gagal CAM {camera_id}: {err}")
            return jsonify({"success": False, "message": err}), 400

        # Simpan ke DB
        ok = db.save_camera_zone(camera_id, mid, near)
        if not ok:
            return jsonify({
                "success": False,
                "message": "Gagal menyimpan zona ke database."
            }), 500

        # Reload ke memory
        try:
            from services.stream_ai_service import reload_camera_zone
            reload_camera_zone(camera_id, mid, near)
            print(f"[ZONES] CAM {camera_id} zona diperbarui dari web")
        except Exception as exc:
            print(f"[ZONES] Gagal reload runtime: {exc}")
            return jsonify({
                "success": True,
                "message": "Zona tersimpan di database, tapi gagal reload runtime. Restart server.",
                "warning": str(exc)
            })

        return jsonify({
            "success": True,
            "message": "Zona berhasil disimpan",
            "data": {
                "camera_id": camera_id,
                "mid": mid,
                "near": near,
            }
        })

    except Exception as exc:
        print(f"[ZONES POST ERROR] {exc}")
        return jsonify({"success": False, "message": str(exc)}), 500


@zones_bp.route("/api/zones/<int:camera_id>/reset", methods=["POST"])
def reset_zone(camera_id):
    """Reset zona ke default."""
    try:
        db.delete_camera_zone(camera_id)

        try:
            from services.stream_ai_service import reload_camera_zone
            # ✅ FIX: pakai DEFAULT baru (y=0.10 / y=0.20)
            reload_camera_zone(camera_id, DEFAULT_MID, DEFAULT_NEAR)
            print(f"[ZONES] CAM {camera_id} zona direset ke default")
        except Exception as exc:
            print(f"[ZONES] Gagal reset runtime: {exc}")

        return jsonify({
            "success": True,
            "message": "Zona direset ke default",
            "data": {
                "camera_id": camera_id,
                "mid": DEFAULT_MID,
                "near": DEFAULT_NEAR,
                "is_custom": False,
            }
        })

    except Exception as exc:
        print(f"[ZONES RESET ERROR] {exc}")
        return jsonify({"success": False, "message": str(exc)}), 500


@zones_bp.route("/api/zones/debug", methods=["GET"])
def debug_zones():
    """Debug: lihat zona aktif di runtime."""
    try:
        from services.stream_ai_service import (
            _runtime_zone_override, CAMERA_ZONE_CONFIG,
            DEFAULT_MID_ZONE, DEFAULT_NEAR_ZONE
        )
        return jsonify({
            "success": True,
            "runtime_override": {
                str(k): {
                    "mid": v.get("mid"),
                    "near": v.get("near"),
                } for k, v in _runtime_zone_override.items()
            },
            "config_keys": list(CAMERA_ZONE_CONFIG.keys()),
            "default_mid": DEFAULT_MID_ZONE,
            "default_near": DEFAULT_NEAR_ZONE,
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500