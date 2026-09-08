from ultralytics import YOLO


class PlateDetector:

    def __init__(
        self,
        model_path="models/plate/model.pt",
        confidence=0.25,
        imgsz=640
    ):

        self.model = YOLO(model_path)

        self.confidence = confidence
        self.imgsz = imgsz

        print("[PLATE] Model berhasil dimuat")
        print("[PLATE] Classes:", self.model.names)
        print("[PLATE] Confidence:", self.confidence)
        print("[PLATE] Image size:", self.imgsz)

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
                device="cpu",
                max_det=10,
                iou=0.7,
                verbose=False,
                stream=False
            )

        except Exception as e:

            print(f"[YOLO ERROR] {e}")

            return []

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            if len(result.boxes) == 0:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                if isinstance(
                    self.model.names,
                    dict
                ):

                    class_name = (
                        self.model.names.get(
                            class_id,
                            str(class_id)
                        )
                    )

                else:

                    class_name = (
                        self.model.names[
                            class_id
                        ]
                    )

                detections.append({

                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ],

                    "confidence": confidence,

                    "class_id": class_id,

                    "class_name": class_name
                })

        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return detections