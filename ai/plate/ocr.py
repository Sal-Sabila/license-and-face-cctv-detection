import cv2
import json
import re
import time
from collections import Counter

from paddleocr import PaddleOCR

try:
    from ai.plate.plate_utils import (
        filter_dan_gabung_spasial,
        koreksi_plat_indonesia,
    )
except Exception:
    filter_dan_gabung_spasial = None
    koreksi_plat_indonesia = None


class PlateOCR:
    """
    OCR plat nomor Indonesia menggunakan PaddleOCR.

    Pipeline:
        crop plat
          -> padding + resize
          -> beberapa preprocessing
          -> PaddleOCR
          -> gabung teks secara spasial
          -> normalisasi
          -> koreksi karakter ambigu
          -> validasi format plat Indonesia
          -> pilih hasil terbaik

    Output:
        {
            "text": "B1234CD",
            "formatted": "B 1234 CD",
            "raw_text": "B1234CD",
            "confidence": 0.82,
            "is_indonesia_pattern": True,
            "variant": "clahe",
            "elapsed": 0.40,
            "variants_tried": 3,
        }
    """

    # OCR confusion yang umum pada plat.
    LETTER_TO_DIGIT = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "T": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }

    DIGIT_TO_LETTER = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
    }

    def __init__(
        self,
        scale=2.5,
        min_confidence=0.15,
        verbose=False,
        fast_mode=True,
    ):
        self.scale = max(float(scale), 2.0)
        self.min_confidence = float(min_confidence)
        self.verbose = bool(verbose)
        self.fast_mode = bool(fast_mode)

        self.min_candidate_length = 3
        self.max_candidate_length = 9

        print("=" * 65)
        print("[OCR] Loading PaddleOCR - Multi Preprocess ALPR")
        print("=" * 65)

        try:
            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except TypeError:
            # Kompatibilitas PaddleOCR versi lama.
            self.ocr = PaddleOCR(lang="en")

        print("[OCR] PaddleOCR siap")
        print(f"[OCR] Scale          : {self.scale}")
        print(f"[OCR] Min confidence : {self.min_confidence}")
        print(f"[OCR] Fast mode      : {self.fast_mode}")
        print("[OCR] Variants       : original + clahe + sharpen + threshold")
        print("=" * 65)

    # ------------------------------------------------------------------
    # BASIC HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    def normalize_text(self, text):
        if text is None:
            return ""
        text = str(text).upper().strip()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    # Alias lama agar kode lain tetap kompatibel.
    _normalize_plate_text = normalize_text

    def _format_plate(self, text):
        text = self.normalize_text(text)
        if not text:
            return ""

        # Cari bentuk: 1-2 huruf + 1-4 angka + 0-3 huruf.
        match = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})$", text)
        if not match:
            return text

        prefix, number, suffix = match.groups()
        return " ".join(x for x in (prefix, number, suffix) if x)

    def is_valid_indonesian_plate(self, text):
        text = self.normalize_text(text)
        if not text:
            return False
        if not (self.min_candidate_length <= len(text) <= self.max_candidate_length):
            return False
        return bool(re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$", text))

    # Alias kompatibilitas.
    _is_valid_indonesian_plate = is_valid_indonesian_plate

    # ------------------------------------------------------------------
    # IMAGE PREPARATION
    # ------------------------------------------------------------------

    def prepare_image(self, image):
        """
        Menyiapkan crop plat tanpa mengubah aspect ratio.
        Ukuran dibatasi supaya OCR CPU tetap masuk akal.
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

            # Padding kecil supaya karakter tepi tidak terpotong.
            pad_y = max(2, int(h * 0.08))
            pad_x = max(3, int(w * 0.06))

            padded = cv2.copyMakeBorder(
                image,
                pad_y,
                pad_y,
                pad_x,
                pad_x,
                cv2.BORDER_REPLICATE,
            )

            ph, pw = padded.shape[:2]
            target_w = max(1, int(pw * self.scale))
            target_h = max(1, int(ph * self.scale))

            # Plat biasanya lebar. Pertahankan rasio dengan resolusi optimal untuk CPU
            max_w = 320
            max_h = 100
            scale_down = min(
                max_w / max(1, target_w),
                max_h / max(1, target_h),
                1.0,
            )

            new_w = max(1, int(target_w * scale_down))
            new_h = max(1, int(target_h * scale_down))

            return cv2.resize(
                padded,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
            )

        except Exception as exc:
            if self.verbose:
                print(f"[OCR PREPARE ERROR] {exc}")
            return None

    # ------------------------------------------------------------------
    # PREPROCESSING VARIANTS
    # ------------------------------------------------------------------

    def preprocess_original(self, image):
        return image.copy() if image is not None else None

    def preprocess_clahe(self, image):
        if image is None or image.size == 0:
            return None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            if self.verbose:
                print(f"[OCR CLAHE ERROR] {exc}")
            return None

    def preprocess_sharpen(self, image):
        if image is None or image.size == 0:
            return None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
            sharp = cv2.addWeighted(gray, 1.45, blur, -0.45, 0)
            return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            if self.verbose:
                print(f"[OCR SHARPEN ERROR] {exc}")
            return None

    def preprocess_threshold(self, image):
        if image is None or image.size == 0:
            return None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            if self.verbose:
                print(f"[OCR THRESHOLD ERROR] {exc}")
            return None

    def generate_variants(self, prepared):
        """Urutan variant: CLAHE pertama (terbukti terbaik untuk kontras CCTV), kemudian original."""
        variants = [
            ("clahe", self.preprocess_clahe(prepared)),
            ("original", self.preprocess_original(prepared)),
        ]

        if not self.fast_mode:
            variants.append(("sharpen", self.preprocess_sharpen(prepared)))

        return [(name, img) for name, img in variants if img is not None]

    # ------------------------------------------------------------------
    # PADDLE RESULT EXTRACTION
    # ------------------------------------------------------------------

    def extract_result(self, result):
        texts, boxes, scores = [], [], []

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
                raw = data.json() if callable(data.json) else data.json
                if isinstance(raw, str):
                    raw = json.loads(raw)
                if isinstance(raw, dict):
                    data = raw.get("res", raw)
            except Exception:
                pass

        if isinstance(data, dict):
            raw_texts = data.get("rec_texts", []) or []
            raw_scores = data.get("rec_scores", []) or []
            raw_boxes = data.get("rec_boxes", []) or []

            if not raw_boxes and data.get("dt_polys") is not None:
                raw_boxes = []
                for poly in data.get("dt_polys") or []:
                    try:
                        xs = [float(p[0]) for p in poly]
                        ys = [float(p[1]) for p in poly]
                        raw_boxes.append([min(xs), min(ys), max(xs), max(ys)])
                    except Exception:
                        raw_boxes.append([0, 0, 0, 0])

            for i, txt in enumerate(raw_texts):
                score = self._safe_float(
                    raw_scores[i] if i < len(raw_scores) else 0.0
                )
                box = raw_boxes[i] if i < len(raw_boxes) else [0, 0, 0, 0]
                if hasattr(box, "tolist"):
                    box = box.tolist()
                texts.append(str(txt))
                boxes.append(box)
                scores.append(score)

            if texts:
                return texts, boxes, scores

            if "text" in data:
                texts.append(str(data.get("text") or ""))
                scores.append(
                    self._safe_float(
                        data.get("confidence", data.get("score", 0.0))
                    )
                )
                boxes.append([0, 0, 0, 0])
                return texts, boxes, scores

        if hasattr(data, "rec_texts"):
            raw_texts = getattr(data, "rec_texts", []) or []
            raw_scores = getattr(data, "rec_scores", []) or []
            raw_boxes = getattr(data, "rec_boxes", []) or []

            for i, txt in enumerate(raw_texts):
                score = self._safe_float(
                    raw_scores[i] if i < len(raw_scores) else 0.0
                )
                box = raw_boxes[i] if i < len(raw_boxes) else [0, 0, 0, 0]
                if hasattr(box, "tolist"):
                    box = box.tolist()
                texts.append(str(txt))
                boxes.append(box)
                scores.append(score)

        return texts, boxes, scores

    # ------------------------------------------------------------------
    # OCR ONE VARIANT
    # ------------------------------------------------------------------

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

            combined = None
            if filter_dan_gabung_spasial is not None:
                try:
                    combined = filter_dan_gabung_spasial(
                        texts,
                        boxes,
                        scores,
                        min_confidence=self.min_confidence,
                    )
                except Exception as exc:
                    if self.verbose:
                        print(f"[OCR SPATIAL WARNING] {exc}")

            # Fallback sederhana jika helper project gagal/tidak ada.
            if combined is None and texts:
                valid_items = []
                for txt, score in zip(texts, scores):
                    clean = self.normalize_text(txt)
                    if clean and score >= self.min_confidence:
                        valid_items.append((clean, score))
                if valid_items:
                    valid_items.sort(key=lambda x: x[1], reverse=True)
                    joined = "".join(x[0] for x in valid_items)
                    avg_score = sum(x[1] for x in valid_items) / len(valid_items)
                    combined = {
                        "text": joined,
                        "formatted": joined,
                        "confidence": avg_score,
                    }

            if combined is None:
                return None

            raw_text = combined.get("text", "")
            confidence = self._safe_float(combined.get("confidence", 0.0))
            elapsed = time.perf_counter() - started

            return {
                "text": self.normalize_text(raw_text),
                "formatted": combined.get("formatted", raw_text),
                "raw_text": str(raw_text or ""),
                "confidence": confidence,
                "is_indonesia_pattern": bool(
                    combined.get("is_indonesia_pattern", False)
                ),
                "variant": variant,
                "elapsed": elapsed,
            }

        except Exception as exc:
            if self.verbose:
                print(f"[OCR ERROR] variant={variant}: {exc}")
            return None

    # ------------------------------------------------------------------
    # CHARACTER CORRECTION
    # ------------------------------------------------------------------

    def _correct_segment(self, segment, target):
        if target == "digit":
            return "".join(self.LETTER_TO_DIGIT.get(ch, ch) for ch in segment)
        return "".join(self.DIGIT_TO_LETTER.get(ch, ch) for ch in segment)

    def _apply_external_correction(self, text):
        if not text or koreksi_plat_indonesia is None:
            return text
        try:
            corrected = koreksi_plat_indonesia(text)
            if isinstance(corrected, str) and corrected.strip():
                return self.normalize_text(corrected)
            if isinstance(corrected, dict):
                value = (
                    corrected.get("formatted")
                    or corrected.get("text")
                    or corrected.get("plate")
                )
                if value:
                    return self.normalize_text(value)
        except Exception:
            pass
        return text

    def correct_plate_text(self, text, confidence=0.0):
        """
        Koreksi konservatif berdasarkan pola Indonesia.

        Tidak mengubah karakter yang tidak ambigu. Koreksi hanya dilakukan
        pada posisi prefix/angka/suffix yang paling masuk akal.
        """
        raw = self.normalize_text(text)
        if not raw:
            return {
                "text": "",
                "formatted": "",
                "correction_applied": False,
                "correction_score": 0.0,
            }

        external = self._apply_external_correction(raw)
        if external and self.is_valid_indonesian_plate(external):
            return {
                "text": external,
                "formatted": self._format_plate(external),
                "correction_applied": external != raw,
                "correction_score": 1.0,
            }

        candidates = []
        n = len(raw)

        # Coba seluruh pemisahan prefix 1-2, angka 1-4, suffix 0-3.
        for prefix_len in (1, 2):
            for digit_len in range(1, 5):
                suffix_len = n - prefix_len - digit_len
                if suffix_len < 0 or suffix_len > 3:
                    continue

                prefix = raw[:prefix_len]
                number = raw[prefix_len:prefix_len + digit_len]
                suffix = raw[prefix_len + digit_len:]

                # Koreksi hanya karakter ambigu di segmen yang seharusnya
                # berupa angka/huruf.
                prefix_c = self._correct_segment(prefix, "letter")
                number_c = self._correct_segment(number, "digit")
                suffix_c = self._correct_segment(suffix, "letter")
                candidate = prefix_c + number_c + suffix_c

                if not self.is_valid_indonesian_plate(candidate):
                    continue

                changes = sum(a != b for a, b in zip(raw, candidate))
                # Penalti perubahan agar koreksi tidak terlalu agresif.
                score = 1.0 - min(changes * 0.12, 0.60)

                # Panjang umum 5-8 karakter mendapat sedikit bonus.
                if 5 <= len(candidate) <= 8:
                    score += 0.05

                candidates.append((score, candidate, changes))

        if not candidates:
            return {
                "text": raw,
                "formatted": self._format_plate(raw),
                "correction_applied": False,
                "correction_score": 0.0,
            }

        candidates.sort(key=lambda x: (x[0], -x[2]), reverse=True)
        _, best, changes = candidates[0]

        return {
            "text": best,
            "formatted": self._format_plate(best),
            "correction_applied": best != raw,
            "correction_score": max(0.0, min(1.0, 1.0 - changes * 0.12)),
        }

    # ------------------------------------------------------------------
    # SCORE RESULT
    # ------------------------------------------------------------------

    def _score_result(self, result):
        if result is None:
            return -1.0

        text = self.normalize_text(result.get("text", ""))
        conf = self._safe_float(result.get("confidence", 0.0))
        valid = self.is_valid_indonesian_plate(text)
        indonesia_flag = bool(result.get("is_indonesia_pattern", False))

        score = conf
        if valid:
            score += 0.35
        elif indonesia_flag:
            score += 0.15

        if 5 <= len(text) <= 8:
            score += 0.03

        return score

    # ------------------------------------------------------------------
    # WARMUP
    # ------------------------------------------------------------------

    def warmup(self):
        try:
            dummy = 255 * __import__("numpy").ones(
                (64, 180, 3), dtype="uint8"
            )
            self._run_ocr(dummy, "warmup")
            print("[OCR] Warm-up selesai")
        except Exception as exc:
            if self.verbose:
                print(f"[OCR WARMUP WARNING] {exc}")

    # ------------------------------------------------------------------
    # PUBLIC READ
    # ------------------------------------------------------------------

    def read(self, image):
        total_start = time.perf_counter()

        empty = {
            "text": "",
            "formatted": "",
            "raw_text": "",
            "confidence": 0.0,
            "is_indonesia_pattern": False,
            "variant": "none",
            "elapsed": 0.0,
            "variants_tried": 0,
            "correction_applied": False,
        }

        if image is None or image.size == 0:
            return empty

        prepared = self.prepare_image(image)
        if prepared is None:
            empty["elapsed"] = time.perf_counter() - total_start
            return empty

        candidates = []
        variants = self.generate_variants(prepared)

        # Jalankan semua variant sampai menemukan hasil yang sangat kuat.
        for variant_name, variant_image in variants:
            result = self._run_ocr(variant_image, variant_name)
            if result is None:
                continue

            correction = self.correct_plate_text(
                result.get("text", ""),
                result.get("confidence", 0.0),
            )

            corrected_text = correction["text"]
            result["raw_text"] = result.get("raw_text") or result.get("text", "")
            result["text"] = corrected_text
            result["formatted"] = correction["formatted"] or corrected_text
            result["is_indonesia_pattern"] = self.is_valid_indonesian_plate(
                corrected_text
            )
            result["correction_applied"] = correction["correction_applied"]
            result["correction_score"] = correction["correction_score"]

            candidates.append(result)

            if self.verbose:
                print(
                    f"[OCR CANDIDATE] {variant_name} -> "
                    f"{result['formatted']} | "
                    f"conf={result['confidence']:.3f} | "
                    f"valid={result['is_indonesia_pattern']}"
                )

            # Early stop cerdas: jika format plat Indonesia valid dan confidence mencukupi (>=0.55),
            # atau confidence sangat tinggi (>=0.70), hentikan segera untuk menghemat CPU.
            if (
                result["is_indonesia_pattern"]
                and result["confidence"] >= 0.55
            ) or (result["confidence"] >= 0.70):
                break

            # Fast mode: setelah original+clahe+sharpen, threshold menjadi
            # fallback terakhir. Tetap dicoba jika hasil belum kuat.

        if not candidates:
            empty["elapsed"] = time.perf_counter() - total_start
            return empty

        # Pilih hasil terbaik berdasarkan confidence + validitas pola.
        best = max(candidates, key=self._score_result)

        # Jika ada beberapa variant dengan teks sama, naikkan confidence
        # sedikit karena hasil konsisten lintas preprocessing.
        normalized_best = self.normalize_text(best.get("text", ""))
        same_text = [
            r for r in candidates
            if self.normalize_text(r.get("text", "")) == normalized_best
        ]
        if len(same_text) >= 2:
            best = dict(best)
            best["confidence"] = min(
                0.99,
                max(
                    best["confidence"],
                    sum(self._safe_float(r.get("confidence", 0.0)) for r in same_text)
                    / len(same_text)
                    + 0.05,
                ),
            )
            best["cross_variant_votes"] = len(same_text)
        else:
            best["cross_variant_votes"] = 1

        best["elapsed"] = time.perf_counter() - total_start
        best["variants_tried"] = len(candidates)
        best["formatted"] = self._format_plate(best.get("text", ""))
        best["text"] = self.normalize_text(best.get("text", ""))
        best["is_indonesia_pattern"] = self.is_valid_indonesian_plate(
            best["text"]
        )

        if self.verbose:
            print(
                f"[OCR BEST] {best['formatted']} | "
                f"conf={best['confidence']:.3f} | "
                f"variant={best['variant']} | "
                f"votes={best.get('cross_variant_votes', 1)} | "
                f"time={best['elapsed']:.2f}s"
            )

        return best