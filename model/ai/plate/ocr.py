import cv2
import re
import json
import time
from paddleocr import PaddleOCR
from ai.plate.plate_utils import filter_dan_gabung_spasial, koreksi_plat_indonesia


class PlateOCR:
    """
    OCR plat nomor menggunakan PaddleOCR.

    Output utama:
        {
            "text": "...",
            "formatted": "...",
            "confidence": 0.0,
            "is_indonesia_pattern": True/False,
            "variant": "original"/"clahe",
            "elapsed": 0.0
        }
    """

    def __init__(
        self,
        scale=2.0,
        min_confidence=0.20,
        verbose=False,
        fast_mode=True,
    ):
        self.scale = scale
        self.min_confidence = min_confidence
        self.verbose = verbose
        self.fast_mode = fast_mode

        # Batas panjang karakter plat yang wajar
        self.min_candidate_length = 3
        self.max_candidate_length = 12

        print("=" * 60)
        print("[OCR] Loading PaddleOCR (Optimized ALPR)...")
        print("=" * 60)

        try:
            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except TypeError:
            # Kompatibilitas dengan versi PaddleOCR lama
            self.ocr = PaddleOCR(lang="en")

        print("[OCR] PaddleOCR siap digunakan")
        print(f"[OCR] Scale           : {self.scale}")
        print(f"[OCR] Min confidence  : {self.min_confidence}")
        print(f"[OCR] Fast mode       : {self.fast_mode}")
        print("=" * 60)

    # ============================================================
    # PREPARE IMAGE
    # ============================================================

    def prepare_image(self, image):
        """
        Persiapan crop plat:
        - Mempertahankan aspect ratio.
        - Padding tipis agar karakter tepi tidak terpotong.
        - Resize proporsional.
        """
        if image is None or image.size == 0:
            return None

        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            h, w = image.shape[:2]

            if h <= 0 or w <= 0:
                return None

            pad_y = max(2, int(h * 0.08))
            pad_x = max(3, int(w * 0.08))

            padded = cv2.copyMakeBorder(
                image,
                pad_y,
                pad_y,
                pad_x,
                pad_x,
                cv2.BORDER_REPLICATE,
            )

            ph, pw = padded.shape[:2]

            scale = max(self.scale, 2.0)
            target_w = int(pw * scale)
            target_h = int(ph * scale)

            # Batas dimensi agar inference tetap ringan
            max_w = 320
            max_h = 128

            scale_down = min(
                max_w / max(1, target_w),
                max_h / max(1, target_h),
                1.0,
            )

            new_w = max(1, int(target_w * scale_down))
            new_h = max(1, int(target_h * scale_down))

            resized = cv2.resize(
                padded,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC,
            )

            return resized

        except Exception as e:
            if self.verbose:
                print(f"[OCR PREPARE ERROR] {e}")
            return None

    # ============================================================
    # PREPROCESSING VARIANT
    # ============================================================

    def preprocess_clahe(self, image):
        """
        CLAHE + contrast enhancement ringan.
        Digunakan sebagai fallback untuk plat yang kurang jelas.
        """
        if image is None or image.size == 0:
            return None

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(6, 6),
            )

            enhanced = clahe.apply(gray)

            blur = cv2.GaussianBlur(
                enhanced,
                (0, 0),
                1.0,
            )

            sharp = cv2.addWeighted(
                enhanced,
                1.30,
                blur,
                -0.30,
                0,
            )

            return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

        except Exception as e:
            if self.verbose:
                print(f"[OCR CLAHE ERROR] {e}")
            return None

    # ============================================================
    # EXTRACT RESULT FROM PADDLE
    # ============================================================

    def extract_result(self, result):
        """
        Mengekstrak texts, boxes, dan scores dari berbagai format
        kembalian PaddleOCR / PaddleX.
        """
        texts = []
        boxes = []
        scores = []

        if result is None:
            return texts, boxes, scores

        if isinstance(result, (list, tuple)):
            for item in result:
                t, b, s = self.extract_result(item)
                texts.extend(t)
                boxes.extend(b)
                scores.extend(s)

            return texts, boxes, scores

        data = result

        if hasattr(data, "json"):
            try:
                j = data.json() if callable(data.json) else data.json

                if isinstance(j, str):
                    j = json.loads(j)

                if isinstance(j, dict):
                    data = j.get("res", j)

            except Exception:
                pass

        # Format PaddleX dictionary
        if isinstance(data, dict):
            raw_texts = data.get("rec_texts", [])
            raw_scores = data.get("rec_scores", [])
            raw_boxes = data.get("rec_boxes", [])

            if (
                raw_boxes is None
                or len(raw_boxes) == 0
            ) and "dt_polys" in data:

                raw_polys = data.get("dt_polys", [])
                boxes_from_poly = []

                for poly in raw_polys:
                    try:
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]

                        boxes_from_poly.append([
                            min(xs),
                            min(ys),
                            max(xs),
                            max(ys),
                        ])

                    except Exception:
                        boxes_from_poly.append([
                            0, 0, 0, 0
                        ])

                raw_boxes = boxes_from_poly

            for i, txt in enumerate(raw_texts or []):
                score = (
                    float(raw_scores[i])
                    if raw_scores is not None
                    and i < len(raw_scores)
                    else 0.0
                )

                box = (
                    raw_boxes[i]
                    if raw_boxes is not None
                    and i < len(raw_boxes)
                    else [0, 0, 0, 0]
                )

                if hasattr(box, "tolist"):
                    box = box.tolist()

                texts.append(str(txt))
                boxes.append(box)
                scores.append(score)

            if texts:
                return texts, boxes, scores

            # Format teks tunggal generic
            if "text" in data:
                texts.append(str(data["text"]))
                scores.append(
                    float(
                        data.get(
                            "confidence",
                            data.get("score", 0.0),
                        )
                    )
                )
                boxes.append([0, 0, 0, 0])

                return texts, boxes, scores

        # Format objek lama PaddleOCR
        if hasattr(data, "rec_texts"):
            raw_texts = getattr(data, "rec_texts", [])
            raw_scores = getattr(data, "rec_scores", [])
            raw_boxes = getattr(data, "rec_boxes", [])

            for i, txt in enumerate(raw_texts or []):
                score = (
                    float(raw_scores[i])
                    if raw_scores is not None
                    and i < len(raw_scores)
                    else 0.0
                )

                box = (
                    raw_boxes[i]
                    if raw_boxes is not None
                    and i < len(raw_boxes)
                    else [0, 0, 0, 0]
                )

                if hasattr(box, "tolist"):
                    box = box.tolist()

                texts.append(str(txt))
                boxes.append(box)
                scores.append(score)

        return texts, boxes, scores

    # ============================================================
    # INFERENCE ONE VARIANT
    # ============================================================

    def _run_ocr(self, image, variant="original"):
        if image is None or image.size == 0:
            return None

        try:
            started = time.perf_counter()

            if hasattr(self.ocr, "predict"):
                results = self.ocr.predict(image)
            else:
                results = self.ocr.ocr(image)

            texts, boxes, scores = self.extract_result(results)

            if self.verbose:
                print(
                    f"[OCR RAW] variant={variant} "
                    f"texts={texts} scores={scores}"
                )

            combined = filter_dan_gabung_spasial(
                texts,
                boxes,
                scores,
                min_confidence=self.min_confidence,
            )

            if combined is None:
                return None

            elapsed = time.perf_counter() - started

            return {
                "text": combined["text"],
                "formatted": combined.get(
                    "formatted",
                    combined["text"],
                ),
                "confidence": float(
                    combined["confidence"]
                ),
                "is_indonesia_pattern": combined.get(
                    "is_indonesia_pattern",
                    False,
                ),
                "variant": variant,
                "elapsed": elapsed,
            }

        except Exception as e:
            if self.verbose:
                print(
                    f"[OCR ERROR] variant={variant}: {e}"
                )

            return None

    # ============================================================
    # WARMUP
    # ============================================================

    def warmup(self):
        """Warm-up awal untuk menginisialisasi model."""
        try:
            dummy = (
                255
                * cv2.UMat(
                    64,
                    160,
                    cv2.CV_8UC3,
                ).get()
            ).astype("uint8")

            self._run_ocr(dummy, "warmup")
            print("[OCR] Warm-up selesai")

        except Exception as exc:
            if self.verbose:
                print(
                    f"[OCR WARMUP WARNING] {exc}"
                )

    # ============================================================
    # READ
    # ============================================================

    def read(self, image):
        """
        Pipeline OCR:
        1. Prepare crop.
        2. OCR original.
        3. Early return jika confidence tinggi / pola Indonesia.
        4. Fallback CLAHE.
        5. Pilih hasil terbaik.
        """
        total_start = time.perf_counter()

        if image is None or image.size == 0:
            return {
                "text": "",
                "formatted": "",
                "confidence": 0.0,
                "is_indonesia_pattern": False,
                "elapsed": 0.0,
            }

        prepared = self.prepare_image(image)

        if prepared is None:
            return {
                "text": "",
                "formatted": "",
                "confidence": 0.0,
                "is_indonesia_pattern": False,
                "elapsed": time.perf_counter()
                - total_start,
            }

        # --------------------------------------------------------
        # PASS 1: ORIGINAL
        # --------------------------------------------------------

        result = self._run_ocr(
            prepared,
            "original",
        )

        if result is not None:
            conf = result["confidence"]
            is_valid_id = result.get(
                "is_indonesia_pattern",
                False,
            )

            if (
                (is_valid_id and conf >= 0.50)
                or conf >= 0.75
            ):
                result["elapsed"] = (
                    time.perf_counter()
                    - total_start
                )

                if self.verbose:
                    print(
                        f"[OCR FAST] "
                        f"{result['formatted']} | "
                        f"conf={conf:.3f} | "
                        f"time={result['elapsed']:.2f}s"
                    )

                return result

        # --------------------------------------------------------
        # PASS 2: CLAHE FALLBACK
        # --------------------------------------------------------

        clahe_img = self.preprocess_clahe(
            prepared
        )

        result_clahe = self._run_ocr(
            clahe_img,
            "clahe",
        )

        # Pilih hasil terbaik
        best = result

        if result_clahe is not None:
            if best is None:
                best = result_clahe

            else:
                score_best = (
                    best["confidence"]
                    + (
                        0.20
                        if best.get(
                            "is_indonesia_pattern"
                        )
                        else 0.0
                    )
                )

                score_clahe = (
                    result_clahe["confidence"]
                    + (
                        0.20
                        if result_clahe.get(
                            "is_indonesia_pattern"
                        )
                        else 0.0
                    )
                )

                if score_clahe > score_best:
                    best = result_clahe

        total_time = (
            time.perf_counter()
            - total_start
        )

        if best is None:
            return {
                "text": "",
                "formatted": "",
                "confidence": 0.0,
                "is_indonesia_pattern": False,
                "elapsed": total_time,
            }

        best["elapsed"] = total_time

        if self.verbose:
            print(
                f"[OCR BEST] "
                f"{best['formatted']} | "
                f"conf={best['confidence']:.3f} | "
                f"time={total_time:.2f}s"
            )

        return best