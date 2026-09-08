from ultralytics import YOLO
import torch


class PlateDetector:

    def __init__(
        self,
        model_path="models/plate/license-plate-finetune-v2n.pt",
        confidence=0.50,
        imgsz=640,
        min_width=30,
        min_height=8,
        min_aspect_ratio=1.8,
        max_aspect_ratio=7.0,
    ):

        self.model_path = model_path

        self.confidence = float(
            confidence
        )

        self.imgsz = int(
            imgsz
        )

        self.min_width = int(
            min_width
        )

        self.min_height = int(
            min_height
        )

        self.min_aspect_ratio = float(
            min_aspect_ratio
        )

        self.max_aspect_ratio = float(
            max_aspect_ratio
        )

        # ======================================================
        # DEVICE
        # ======================================================

        if torch.cuda.is_available():

            self.device = 0

            print(
                "[PLATE V2] CUDA tersedia."
            )

            print(
                "[PLATE V2] Device: GPU"
            )

        else:

            self.device = "cpu"

            print(
                "[PLATE V2] CUDA tidak tersedia."
            )

            print(
                "[PLATE V2] Device: CPU"
            )

        # ======================================================
        # LOAD MODEL
        # ======================================================

        print("=" * 60)
        print("[PLATE V2] Loading YOLO...")
        print("=" * 60)

        self.model = YOLO(
            model_path
        )

        print(
            "[PLATE V2] Model berhasil dimuat"
        )

        print(
            "[PLATE V2] Model path:",
            model_path
        )

        print(
            "[PLATE V2] Classes:",
            self.model.names
        )

        print(
            "[PLATE V2] Confidence:",
            self.confidence
        )

        print(
            "[PLATE V2] Image size:",
            self.imgsz
        )

        print(
            "[PLATE V2] Device:",
            self.device
        )

        print(
            "[PLATE V2] Min bbox:",
            f"{self.min_width}x"
            f"{self.min_height}"
        )

        print(
            "[PLATE V2] Aspect ratio:",
            f"{self.min_aspect_ratio}-"
            f"{self.max_aspect_ratio}"
        )

        print("=" * 60)

    # ==========================================================
    # VALIDASI BBOX
    # ==========================================================

    def _valid_bbox(
        self,
        bbox,
        frame_shape
    ):

        if bbox is None:
            return False

        if len(bbox) != 4:
            return False

        try:

            x1, y1, x2, y2 = [
                int(v)
                for v in bbox
            ]

        except Exception:

            return False

        frame_h, frame_w = (
            frame_shape[:2]
        )

        # ------------------------------------------------------
        # Batasi koordinat
        # ------------------------------------------------------

        x1 = max(
            0,
            min(x1, frame_w)
        )

        y1 = max(
            0,
            min(y1, frame_h)
        )

        x2 = max(
            0,
            min(x2, frame_w)
        )

        y2 = max(
            0,
            min(y2, frame_h)
        )

        width = x2 - x1
        height = y2 - y1

        # ------------------------------------------------------
        # Ukuran minimum
        # ------------------------------------------------------

        if width < self.min_width:
            return False

        if height < self.min_height:
            return False

        if width <= 0:
            return False

        if height <= 0:
            return False

        # ------------------------------------------------------
        # Aspect ratio
        # ------------------------------------------------------

        aspect_ratio = (
            width / height
        )

        if (
            aspect_ratio
            < self.min_aspect_ratio
        ):
            return False

        if (
            aspect_ratio
            > self.max_aspect_ratio
        ):
            return False

        return True

    # ==========================================================
    # DETECT
    # ==========================================================

    def detect(
        self,
        frame
    ):

        if frame is None:
            return []

        if frame.size == 0:
            return []

        try:

            results = self.model.predict(

                source=frame,

                # ==============================================
                # YOLO THRESHOLD
                # ==============================================

                conf=self.confidence,

                # ==============================================
                # IMAGE SIZE
                # ==============================================

                imgsz=self.imgsz,

                # ==============================================
                # AUTO DEVICE
                # ==============================================

                device=self.device,

                # ==============================================
                # MAX DETECTION
                # ==============================================

                max_det=10,

                # ==============================================
                # NMS IoU
                # ==============================================

                iou=0.7,

                verbose=False,

                stream=False,

            )

        except Exception as e:

            print(
                f"[YOLO V2 ERROR] {e}"
            )

            return []

        detections = []

        # ======================================================
        # PARSE HASIL YOLO
        # ======================================================

        for result in results:

            if result.boxes is None:
                continue

            if len(result.boxes) == 0:
                continue

            for box in result.boxes:

                try:

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                except Exception:

                    continue

                # ==================================================
                # CLASS NAME
                # ==================================================

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

                # ==================================================
                # CLASS FILTER
                # ==================================================

                # Dataset Anda seharusnya:
                # class 0 = license plate

                if class_id != 0:
                    continue

                # ==================================================
                # BBOX
                # ==================================================

                bbox = [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ]

                # ==================================================
                # GEOMETRY FILTER
                # ==================================================

                if not self._valid_bbox(
                    bbox,
                    frame.shape
                ):

                    continue

                # ==================================================
                # SIMPAN DETEKSI
                # ==================================================

                detections.append({

                    "bbox": bbox,

                    "confidence": (
                        confidence
                    ),

                    "class_id": (
                        class_id
                    ),

                    "class_name": (
                        class_name
                    ),

                })

        # ======================================================
        # SORT CONFIDENCE
        # ======================================================

        detections.sort(

            key=lambda x:
                x["confidence"],

            reverse=True

        )

        return detections