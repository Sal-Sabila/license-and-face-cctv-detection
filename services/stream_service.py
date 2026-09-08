from datetime import datetime


class StreamService:
    def __init__(self):
        self.stream_url = None
        self.running = False
        self.cameras = []
        self.next_id = 1

    def list_cameras(self):
        return self.cameras

    def add_camera(self, data):
        name = str(data.get("name", "")).strip()
        url = str(data.get("url", "")).strip()
        if not name or not url:
            return None
        camera = {
            "id": self.next_id,
            "name": name,
            "alias": str(data.get("alias", name)).strip() or name,
            "region": str(data.get("region", "-")).strip() or "-",
            "coordinate": str(data.get("coordinate", "-")).strip() or "-",
            "adm4": str(data.get("adm4", "-")).strip() or "-",
            "url": url,
            "active": bool(data.get("active", True)),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.cameras.append(camera)
        self.next_id += 1
        return camera

    def update_camera(self, camera_id, data):
        camera = self.get_camera(camera_id)
        if not camera:
            return None
        for field in ("name", "alias", "region", "coordinate", "adm4", "url"):
            if field in data:
                value = str(data[field]).strip()
                if field in ("name", "url") and not value:
                    return None
                camera[field] = value
        if "active" in data:
            camera["active"] = bool(data["active"])
        return camera

    def delete_camera(self, camera_id):
        camera = self.get_camera(camera_id)
        if not camera:
            return False
        self.cameras.remove(camera)
        if self.stream_url == camera["url"]:
            self.stop()
        return True

    def get_camera(self, camera_id):
        try:
            camera_id = int(camera_id)
        except (TypeError, ValueError):
            return None
        return next((camera for camera in self.cameras if camera["id"] == camera_id), None)

    def start(self, stream_url):
        if not stream_url:
            return {"success": False, "message": "RTMP URL belum diberikan"}
        self.stream_url = stream_url
        self.running = True
        return {"success": True, "message": "Stream berhasil dimulai", "stream_url": stream_url}

    def stop(self):
        self.running = False
        return {"success": True, "message": "Stream dihentikan"}

    def status(self):
        return {
            "running": self.running,
            "stream_url": self.stream_url,
            "camera_count": len(self.cameras),
            "active_count": sum(camera["active"] for camera in self.cameras)
        }
