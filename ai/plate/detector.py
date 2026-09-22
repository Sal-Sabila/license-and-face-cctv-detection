import os
import cv2
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
        min_height=8,
        min_aspect_ratio=2.0,
        max_aspect_ratio=6.0,
        small_roi_scale=2.0,
        **kwargs
    ):
        # ============================================================
        # SANITASI: pastikan model_path selalu string
        # ============================================================
        if isinstance(model_path, (tuple, list)):
            model_path = model_path[0] if len(model_path) > 0 else ""
        model_path = str(model_path).strip()

        if not os.path.isabs(model_path):
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            candidate = os.path.join(project_root, model_path)
            if os.path.isfile(candidate):
                model_path = candidate

        print(f"[PLATE] model_path = {model_path!r}")
        print(f"[PLATE] file exists = {os.path.isfile(model_path)}")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model plate tidak ditemukan: {model_path}"
            )

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
        self.small_roi_scale = max(1.0, float(small_roi_scale))

        print("=" * 60)
        print("[PLATE] Detector siap")
        print("[PLATE] Model:", model_path)
        print("[PLATE] Classes:", self.model.names)
        print("[PLATE] Confidence:", self.confidence)
        print("[PLATE] Image size:", self.imgsz)
        print("[PLATE] Device:", self.device)
        print("[PLATE] Aspect ratio:", self.min_aspect_ratio, "-", self.max_aspect_ratio)
        print("[PLATE] Small ROI scale:", self.small_roi_scale)
        print("=" * 60)

    def detect(self, frame):
        if frame is None:
            return []
        if frame.size == 0:
            return []

        try:
            original_height, original_width = frame.shape[:2]
            inference_frame = frame
            scale = 1.0

            if max(original_height, original_width) < 640:
                scale = min(
                    self.small_roi_scale,
                    640.0 / max(1, max(original_height, original_width)),
                )
                if scale > 1.05:
                    inference_frame = cv2.resize(
                        frame,
                        (int(original_width * scale), int(original_height * scale)),
                        interpolation=cv2.INTER_CUBIC,
                    )

            results = self.model.predict(
                source=inference_frame,
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

                    if class_id != 0:
                        continue

                    x1, y1, x2, y2 = [value / scale for value in box.xyxy[0].tolist()]
                except Exception:
                    continue

                x1 = max(0, min(int(x1), frame_width - 1))
                y1 = max(0, min(int(y1), frame_height - 1))
                x2 = max(0, min(int(x2), frame_width - 1))
                y2 = max(0, min(int(y2), frame_height - 1))

                width = x2 - x1
                height = y2 - y1

                if width < self.min_width:
                    continue
                if height < self.min_height:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                aspect_ratio = width / max(height, 1)

                if aspect_ratio < self.min_aspect_ratio:
                    continue
                if aspect_ratio > self.max_aspect_ratio:
                    continue

                pad_w = max(2, int(width * 0.08))
                pad_h = max(2, int(height * 0.08))
                cy1 = max(0, y1 - pad_h)
                cy2 = min(frame_height, y2 + pad_h)
                cx1 = max(0, x1 - pad_w)
                cx2 = min(frame_width, x2 + pad_w)
                crop = frame[cy1:cy2, cx1:cx2]

                if crop is None or crop.size == 0:
                    continue

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if isinstance(self.model.names, dict):
                    class_name = self.model.names.get(class_id, str(class_id))
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

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections