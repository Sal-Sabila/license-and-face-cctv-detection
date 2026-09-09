from ultralytics import YOLO


class PlateDetector:

    def __init__(
        self,
        model_path="models/plate/license-plate-finetune-v2n.pt",
        confidence=0.50,
        imgsz=640,
        device="cpu",
        max_det=10,
        iou=0.45,
        min_width=30,
        min_height=10,
        min_aspect_ratio=1.5,
        max_aspect_ratio=6.5
    ):

        self.model = YOLO(model_path)

        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.iou = iou

        self.min_width = min_width
        self.min_height = min_height

        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        print("=" * 60)
        print("[PLATE] Detector siap")
        print("[PLATE] Model:", model_path)
        print("[PLATE] Classes:", self.model.names)
        print("[PLATE] Confidence:", self.confidence)
        print("[PLATE] Image size:", self.imgsz)
        print("[PLATE] Device:", self.device)
        print("=" * 60)

    def detect(self, frame):

        if frame is None:
            return []

        if frame.size == 0:
            return []

        try:

            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_det,
                iou=self.iou,
                verbose=False,
                stream=False
            )

        except Exception as e:

            print(f"[PLATE ERROR] YOLO inference failed: {e}")
            return []

        detections = []

        frame_height, frame_width = frame.shape[:2]

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                try:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Model kamu hanya memiliki class 0 = license_plate
                    if class_id != 0:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                except Exception:
                    continue

                # Clamp koordinat
                x1 = max(0, min(int(x1), frame_width - 1))
                y1 = max(0, min(int(y1), frame_height - 1))
                x2 = max(0, min(int(x2), frame_width - 1))
                y2 = max(0, min(int(y2), frame_height - 1))

                width = x2 - x1
                height = y2 - y1

                # Validasi ukuran
                if width < self.min_width:
                    continue

                if height < self.min_height:
                    continue

                if x2 <= x1 or y2 <= y1:
                    continue

                # Validasi rasio bentuk plat
                aspect_ratio = width / max(height, 1)

                if aspect_ratio < self.min_aspect_ratio:
                    continue

                if aspect_ratio > self.max_aspect_ratio:
                    continue

                # Crop
                crop = frame[y1:y2, x1:x2]

                if crop is None or crop.size == 0:
                    continue

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if isinstance(self.model.names, dict):
                    class_name = self.model.names.get(
                        class_id,
                        str(class_id)
                    )
                else:
                    class_name = self.model.names[class_id]

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "center": [center_x, center_y],
                    "crop": crop
                })

        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return detections