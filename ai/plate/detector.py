from ultralytics import YOLO


class PlateDetector:

    def __init__(
        self,
        model_path="models/plate/license-plate-finetune-v2n.pt",
        confidence=0.25,
        imgsz=640,
        device="cpu",
        max_det=10,
        iou=0.7,
        min_width=20,
        min_height=8
    ):

        self.model = YOLO(model_path)

        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.iou = iou

        self.min_width = min_width
        self.min_height = min_height

        print("[PLATE] Model berhasil dimuat")
        print("[PLATE] Classes:", self.model.names)
        print("[PLATE] Confidence:", self.confidence)
        print("[PLATE] Image size:", self.imgsz)
        print("[PLATE] Device:", self.device)

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

            print(f"[PLATE ERROR] {e}")

            return []

        detections = []

        frame_height, frame_width = frame.shape[:2]

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

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

                # Validasi koordinat
                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop plat
                crop = frame[y1:y2, x1:x2]

                if crop is None or crop.size == 0:
                    continue

                # Center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Nama class
                if isinstance(self.model.names, dict):

                    class_name = self.model.names.get(
                        class_id,
                        str(class_id)
                    )

                else:

                    class_name = self.model.names[class_id]

                detections.append({

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "confidence": confidence,

                    "class_id": class_id,

                    "class_name": class_name,

                    "width": width,

                    "height": height,

                    "center": [
                        center_x,
                        center_y
                    ],

                    "crop": crop
                })

        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return detections