from datetime import datetime


class DetectionService:
    def __init__(self):
        self.latest_result = None
        self.history = []
        self.latest_face_result = None
        self.face_history = []

    def get_latest(self):
        return self.latest_result

    def get_history(self):
        return self.history

    def get_latest_face(self):
        return self.latest_face_result

    def get_face_history(self):
        return self.face_history

    def add_detection(self, plate, confidence, camera, bbox=None):
        result = self._build_result(
            confidence=confidence,
            camera=camera,
            bbox=bbox,
            plate=plate
        )
        self.latest_result = result
        self.history.insert(0, result)
        self.history = self.history[:100]
        return result

    def add_face_detection(self, face_count, confidence, camera, bbox=None):
        result = self._build_result(
            confidence=confidence,
            camera=camera,
            bbox=bbox,
            type="face",
            face_count=int(face_count)
        )
        self.latest_face_result = result
        self.face_history.insert(0, result)
        self.face_history = self.face_history[:100]
        return result

    def _build_result(self, confidence, camera, bbox=None, **data):
        confidence = float(confidence)
        if confidence > 1:
            confidence /= 100
        confidence = max(0, min(1, confidence))
        return {
            **data,
            "confidence": round(confidence, 2),
            "confidence_percent": round(confidence * 100, 1),
            "camera": camera,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bbox": bbox,
            "status": self.get_status(confidence)
        }

    def get_status(self, confidence):
        return "Terbaca" if confidence >= 0.80 else "Perlu cek"

    def clear_history(self):
        self.history = []
        self.latest_result = None

    def clear_face_history(self):
        self.face_history = []
        self.latest_face_result = None
