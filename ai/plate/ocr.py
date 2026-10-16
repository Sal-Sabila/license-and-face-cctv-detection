import glob
import os
import cv2
import json
import re
import time
from collections import Counter
import numpy as np

from paddleocr import PaddleOCR

DEBUG_OCR_DIR = os.path.join("static", "captures", "debug_ocr")


def deskew_plate(image):
    """Koreksi kemiringan sudut plat nomor menggunakan kontur/minAreaRect."""
    if image is None or image.size == 0:
        return image
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 20:
            return image
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle
        if abs(angle) < 1.5 or abs(angle) > 32.0:
            return image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    except Exception:
        return image


def save_debug_crop_pair(
    raw_crop, prep_crop, text, conf, valid, max_keep=50, context=None
):
    """Menyimpan perbandingan crop plat sebelum vs sesudah preprocessing untuk evaluasi."""
    try:
        os.makedirs(DEBUG_OCR_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        stamp = time.time_ns()
        clean_text = re.sub(r"[^A-Z0-9]", "", str(text or "EMPTY").upper())[:10]
        status = "VALID" if valid else "INVALID"
        context_label = re.sub(r"[^A-Z0-9_-]", "", str(context or "").upper())[:40]
        context_suffix = f"_{context_label}" if context_label else ""
        base_name = (
            f"{ts}_{stamp % 1_000_000_000:09d}{context_suffix}_"
            f"{status}_{clean_text}_c{int(conf * 100)}"
        )

        saved_paths = []
        if raw_crop is not None and hasattr(raw_crop, "size") and raw_crop.size > 0:
            raw_path = os.path.join(DEBUG_OCR_DIR, f"{base_name}_1raw.jpg")
            if cv2.imwrite(raw_path, raw_crop):
                saved_paths.append(raw_path)
            else:
                print(f"[OCR DEBUG WARNING] Gagal menulis crop mentah: {raw_path}")
        if prep_crop is not None and hasattr(prep_crop, "size") and prep_crop.size > 0:
            prep_path = os.path.join(DEBUG_OCR_DIR, f"{base_name}_2prep.jpg")
            if cv2.imwrite(prep_path, prep_crop):
                saved_paths.append(prep_path)
            else:
                print(f"[OCR DEBUG WARNING] Gagal menulis crop preprocessing: {prep_path}")

        if saved_paths:
            print(
                f"[OCR DEBUG] Crop sebelum/sesudah preprocessing disimpan: "
                f"{', '.join(saved_paths)}"
            )

        files = sorted(glob.glob(os.path.join(DEBUG_OCR_DIR, "*.jpg")))
        if len(files) > max_keep * 2:
            for old_f in files[:len(files) - max_keep * 2]:
                try:
                    os.remove(old_f)
                except OSError as exc:
                    print(f"[OCR DEBUG WARNING] Gagal menghapus sampel lama {old_f}: {exc}")
    except (OSError, cv2.error) as exc:
        print(f"[OCR DEBUG ERROR] Gagal menyimpan crop OCR: {exc}")

# ✅ IMPORT VALIDATOR BARU
try:
    from ai.plate.validator import (
        correct_and_validate_plate,
        is_valid_plate,
        format_plate_for_display,
        normalize_plate_text,
        VALID_REGION_CODES,
    )
    HAS_PLATE_VALIDATOR = True
except ImportError:
    HAS_PLATE_VALIDATOR = False
    VALID_REGION_CODES = set()
    print("[OCR WARNING] ai.plate.validator tidak tersedia, pakai fallback regex")

# Helper project (opsional, tidak wajib)
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
          -> beberapa preprocessing (clahe/original/sharpen/threshold)
          -> PaddleOCR
          -> gabung teks secara spasial
          -> normalisasi
          -> koreksi karakter ambigu + validasi kode wilayah
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
            "correction_applied": False,
            "region_code": "B",
        }
    """

    # OCR confusion (fallback kalau validator tidak tersedia)
    LETTER_TO_DIGIT = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "J": "1",
        "Z": "2", "S": "5",
        "G": "6", "B": "8",
    }

    DIGIT_TO_LETTER = {
        "0": "O", "1": "I", "2": "Z",
        "5": "S", "6": "G", "8": "B",
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
        print(f"[OCR] Validator      : {'AKTIF (kode wilayah)' if HAS_PLATE_VALIDATOR else 'FALLBACK REGEX'}")
        print(
            "[OCR] Variants       : adaptive + original fallback"
            + (" + sharpen" if not fast_mode else "")
        )
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
        """Normalisasi dasar: uppercase, hapus non-alphanumeric."""
        if text is None:
            return ""
        if HAS_PLATE_VALIDATOR:
            return normalize_plate_text(text)
        text = str(text).upper().strip()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    _normalize_plate_text = normalize_text

    def _format_plate(self, text):
        """Format untuk display: 'AB1234CD' -> 'AB 1234 CD'."""
        if HAS_PLATE_VALIDATOR:
            return format_plate_for_display(text)
        # Fallback
        text = self.normalize_text(text)
        if not text:
            return ""
        match = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})$", text)
        if not match:
            return text
        prefix, number, suffix = match.groups()
        return " ".join(x for x in (prefix, number, suffix) if x)

    def is_valid_indonesian_plate(self, text):
        """
        ✅ Validasi plat Indonesia menggunakan kode wilayah resmi.

        Fallback ke regex longgar kalau validator tidak tersedia.
        """
        if not text:
            return False

        if HAS_PLATE_VALIDATOR:
            return is_valid_plate(text)

        # Fallback regex longgar
        text = self.normalize_text(text)
        if not text:
            return False
        if not (self.min_candidate_length <= len(text) <= self.max_candidate_length):
            return False
        return bool(re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$", text))

    _is_valid_indonesian_plate = is_valid_indonesian_plate

    def _extract_region_code(self, text):
        """Ambil kode wilayah dari plat (kalau valid)."""
        if not text:
            return None
        text = self.normalize_text(text)
        for length in (2, 1):
            if len(text) >= length + 1:
                prefix = text[:length]
                if prefix in VALID_REGION_CODES:
                    return prefix
        return None

    # ------------------------------------------------------------------
    # IMAGE PREPARATION
    # ------------------------------------------------------------------

    def prepare_image(self, image):
        """
        Menyiapkan crop plat:
        1. Koreksi kemiringan (deskew).
        2. Resize proporsional dengan Lanczos-4 ke tinggi optimal pengenalan.
        3. Tambah border padding seragam.
        """
        if image is None or image.size == 0:
            return None

        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # 1. Koreksi kemiringan
            deskewed = deskew_plate(image)
            if deskewed is not None and deskewed.size > 0:
                image = deskewed

            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                return None

            # Upscale tiny detections while preserving their original aspect ratio.
            scale = max(96.0 / float(h), 192.0 / float(w))
            scale = min(scale, 480.0 / float(w))
            target_w = max(1, int(round(w * scale)))
            target_h = max(1, int(round(h * scale)))

            resized = cv2.resize(
                image,
                (target_w, target_h),
                interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_LANCZOS4,
            )

            # 3. Tambahkan border padding bersih
            pad_y = 6
            pad_x = 10
            padded = cv2.copyMakeBorder(
                resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE,
            )
            return padded

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
        """CLAHE pada kanal Luminance (LAB) untuk mempertajam teks tanpa artefak warna."""
        if image is None or image.size == 0:
            return None
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 4))
            l_clahe = clahe.apply(l)
            blurred = cv2.GaussianBlur(l_clahe, (0, 0), 0.8)
            l_clahe = cv2.addWeighted(l_clahe, 1.35, blurred, -0.35, 0)
            merged = cv2.merge([l_clahe, a, b])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception as exc:
            if self.verbose:
                print(f"[OCR CLAHE ERROR] {exc}")
            return None

    def preprocess_sharpen(self, image):
        """Unsharp masking untuk mempertegas garis karakter plat."""
        if image is None or image.size == 0:
            return None
        try:
            clahe_img = self.preprocess_clahe(image)
            base = clahe_img if clahe_img is not None else image
            blur = cv2.GaussianBlur(base, (0, 0), 1.0)
            sharp = cv2.addWeighted(base, 1.5, blur, -0.5, 0)
            return sharp
        except Exception as exc:
            if self.verbose:
                print(f"[OCR SHARPEN ERROR] {exc}")
            return None

    def preprocess_inverted(self, image):
        """Inversi untuk plat berlatar gelap / hitam (TNKB standar lama) agar DBNet dapat mendeteksi teks."""
        if image is None or image.size == 0:
            return None
        try:
            inv = cv2.bitwise_not(image)
            return self.preprocess_clahe(inv)
        except Exception as exc:
            if self.verbose:
                print(f"[OCR INVERT ERROR] {exc}")
            return None

    def preprocess_threshold(self, image):
        if image is None or image.size == 0:
            return None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 7,
            )
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            if self.verbose:
                print(f"[OCR THRESHOLD ERROR] {exc}")
            return None

    def generate_variants(self, prepared):
        """
        Urutan varian cerdas:
        - Jika plat gelap (latar hitam): prioritaskan varian inversi CLAHE agar DBNet deteksi teks putih.
        - Jika plat terang (latar putih): prioritaskan CLAHE lalu original.
        """
        if prepared is None:
            return []

        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        is_dark_bg = bool(np.mean(gray) < 118.0)

        if is_dark_bg:
            primary = self.preprocess_inverted(prepared)
            primary_name = "inv_clahe"
        else:
            primary = self.preprocess_clahe(prepared)
            primary_name = "clahe"

        variants = [(primary_name, primary)]
        if not self.fast_mode:
            variants.extend((
                ("original", self.preprocess_original(prepared)),
                ("sharpen", self.preprocess_sharpen(prepared)),
            ))
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
                clean_txt = re.sub(r"[^\x20-\x7E]", "", str(txt)).strip()
                if clean_txt:
                    texts.append(clean_txt)
                    boxes.append(box)
                    scores.append(score)

            if texts:
                return texts, boxes, scores

            if "text" in data:
                clean_txt = re.sub(r"[^\x20-\x7E]", "", str(data.get("text") or "")).strip()
                if clean_txt:
                    texts.append(clean_txt)
                    scores.append(
                        self._safe_float(data.get("confidence", data.get("score", 0.0)))
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
                clean_txt = re.sub(r"[^\x20-\x7E]", "", str(txt)).strip()
                if clean_txt:
                    texts.append(clean_txt)
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

            paddle_started = time.perf_counter()
            if hasattr(self.ocr, "predict"):
                results = self.ocr.predict(image)
            else:
                results = self.ocr.ocr(image)
            paddle_inference_elapsed = time.perf_counter() - paddle_started

            texts, boxes, scores = self.extract_result(results)

            if self.verbose:
                print(f"[OCR RAW] variant={variant} texts={texts} scores={scores}")

            combined = None
            if filter_dan_gabung_spasial is not None:
                try:
                    combined = filter_dan_gabung_spasial(
                        texts, boxes, scores,
                        min_confidence=self.min_confidence,
                    )
                except Exception as exc:
                    if self.verbose:
                        print(f"[OCR SPATIAL WARNING] {exc}")

            # Fallback sederhana
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
                return {
                    "text": "",
                    "paddle_inference_elapsed": paddle_inference_elapsed,
                    "no_text": True,
                }

            raw_text = combined.get("text", "")
            confidence = self._safe_float(combined.get("confidence", 0.0))
            elapsed = time.perf_counter() - started

            return {
                "text": self.normalize_text(raw_text),
                "formatted": combined.get("formatted", raw_text),
                "raw_text": str(raw_text or ""),  # ✅ raw untuk audit
                "confidence": confidence,
                "is_indonesia_pattern": False,  # akan di-set setelah koreksi
                "variant": variant,
                "elapsed": elapsed,
                "paddle_inference_elapsed": paddle_inference_elapsed,
            }

        except Exception as exc:
            if self.verbose:
                print(f"[OCR ERROR] variant={variant}: {exc}")
            return None

    # ------------------------------------------------------------------
    # CHARACTER CORRECTION
    # ------------------------------------------------------------------

    def _correct_segment(self, segment, target):
        """Fallback koreksi karakter per segmen (kalau validator tidak ada)."""
        if target == "digit":
            return "".join(self.LETTER_TO_DIGIT.get(ch, ch) for ch in segment)
        return "".join(self.DIGIT_TO_LETTER.get(ch, ch) for ch in segment)

    def _apply_external_correction(self, text):
        """Pakai helper project kalau ada."""
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
        ✅ Koreksi & validasi menggunakan kode wilayah Indonesia.

        Return dict:
            {
                "text": str,           # teks terkoreksi
                "formatted": str,      # untuk display
                "correction_applied": bool,
                "correction_score": float,
                "region_code": str|None,
            }
        """
        raw = self.normalize_text(text)
        if not raw:
            return {
                "text": "", "formatted": "",
                "correction_applied": False, "correction_score": 0.0,
                "region_code": None,
            }

        # PRIORITAS 1: Pakai validator baru
        if HAS_PLATE_VALIDATOR:
            result = correct_and_validate_plate(raw, confidence)

            if result["valid"]:
                return {
                    "text": result["corrected"],
                    "formatted": format_plate_for_display(result["corrected"]),
                    "correction_applied": result["corrected"] != raw,
                    "correction_score": 1.0 - result["confidence_penalty"],
                    "region_code": self._extract_region_code(result["corrected"]),
                }
            else:
                # Tidak valid → kembalikan raw (untuk review)
                return {
                    "text": raw,
                    "formatted": self._format_plate(raw),
                    "correction_applied": False,
                    "correction_score": 0.0,
                    "region_code": None,
                }

        # FALLBACK: Pakai helper project
        external = self._apply_external_correction(raw)
        if external and self.is_valid_indonesian_plate(external):
            return {
                "text": external,
                "formatted": self._format_plate(external),
                "correction_applied": external != raw,
                "correction_score": 1.0,
                "region_code": None,
            }

        # FALLBACK: Koreksi manual per segmen
        candidates = []
        n = len(raw)

        for prefix_len in (1, 2):
            for digit_len in range(1, 5):
                suffix_len = n - prefix_len - digit_len
                if suffix_len < 0 or suffix_len > 3:
                    continue

                prefix = raw[:prefix_len]
                number = raw[prefix_len:prefix_len + digit_len]
                suffix = raw[prefix_len + digit_len:]

                prefix_c = self._correct_segment(prefix, "letter")
                number_c = self._correct_segment(number, "digit")
                suffix_c = self._correct_segment(suffix, "letter")
                candidate = prefix_c + number_c + suffix_c

                if not self.is_valid_indonesian_plate(candidate):
                    continue

                changes = sum(a != b for a, b in zip(raw, candidate))
                score = 1.0 - min(changes * 0.12, 0.60)
                if 5 <= len(candidate) <= 8:
                    score += 0.05

                candidates.append((score, candidate, changes))

        if not candidates:
            return {
                "text": raw,
                "formatted": self._format_plate(raw),
                "correction_applied": False,
                "correction_score": 0.0,
                "region_code": None,
            }

        candidates.sort(key=lambda x: (x[0], -x[2]), reverse=True)
        _, best, changes = candidates[0]

        return {
            "text": best,
            "formatted": self._format_plate(best),
            "correction_applied": best != raw,
            "correction_score": max(0.0, min(1.0, 1.0 - changes * 0.12)),
            "region_code": self._extract_region_code(best),
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
        region = self._extract_region_code(text)

        score = conf
        if valid:
            score += 0.35
            # Bonus kalau kode wilayah dikenal
            if region:
                score += 0.10
        elif indonesia_flag:
            score += 0.15

        if 5 <= len(text) <= 8:
            score += 0.03

        # Bonus kalau tidak ada koreksi (raw == text)
        if not result.get("correction_applied", False):
            score += 0.05

        return score

    # ------------------------------------------------------------------
    # WARMUP
    # ------------------------------------------------------------------

    def warmup(self):
        try:
            import numpy as np
            dummy = 255 * np.ones((64, 180, 3), dtype="uint8")
            self._run_ocr(dummy, "warmup")
            print("[OCR] Warm-up selesai")
        except Exception as exc:
            if self.verbose:
                print(f"[OCR WARMUP WARNING] {exc}")

    # ------------------------------------------------------------------
    # PUBLIC READ
    # ------------------------------------------------------------------

    def read(self, image, context=None):
        total_start = time.perf_counter()

        empty = {
            "text": "",
            "formatted": "",
            "raw_text": "",
            "confidence": 0.0,
            "is_indonesia_pattern": False,
            "variant": "none",
            "elapsed": 0.0,
            "paddle_inference_elapsed": 0.0,
            "variants_tried": 0,
            "correction_applied": False,
            "region_code": None,
        }

        if image is None or image.size == 0:
            return empty

        prepared = self.prepare_image(image)
        if prepared is None:
            empty["elapsed"] = time.perf_counter() - total_start
            return empty

        candidates = []
        paddle_inference_elapsed = 0.0
        attempted_variants = 0

        def read_variant(variant_name, variant_image):
            nonlocal paddle_inference_elapsed, attempted_variants
            attempted_variants += 1
            variant_started = time.perf_counter()
            result = self._run_ocr(variant_image, variant_name)
            variant_elapsed = time.perf_counter() - variant_started
            paddle_inference_elapsed += (
                self._safe_float(result.get("paddle_inference_elapsed"), variant_elapsed)
                if result is not None
                else variant_elapsed
            )
            if result is None or result.get("no_text"):
                return None

            # ✅ KOREKSI + VALIDASI
            correction = self.correct_plate_text(
                result.get("text", ""),
                result.get("confidence", 0.0),
            )

            corrected_text = correction["text"]
            result["raw_text"] = result.get("raw_text") or result.get("text", "")
            result["text"] = corrected_text
            result["formatted"] = correction["formatted"] or corrected_text
            result["is_indonesia_pattern"] = self.is_valid_indonesian_plate(corrected_text)
            result["correction_applied"] = correction["correction_applied"]
            result["correction_score"] = correction["correction_score"]
            result["region_code"] = correction["region_code"]

            if self.verbose:
                print(
                    f"[OCR CANDIDATE] {variant_name} -> "
                    f"{result['formatted']} | "
                    f"raw={result['raw_text']} | "
                    f"conf={result['confidence']:.3f} | "
                    f"valid={result['is_indonesia_pattern']} | "
                    f"region={result['region_code']}"
                )

            return result

        for variant_name, variant_image in self.generate_variants(prepared):
            result = read_variant(variant_name, variant_image)
            if result is not None:
                candidates.append(result)
                if (
                    result["is_indonesia_pattern"]
                    and result["confidence"] >= 0.70
                ):
                    break

        has_strong_valid_read = any(
            result["is_indonesia_pattern"] and result["confidence"] >= 0.70
            for result in candidates
        )
        if self.fast_mode and not has_strong_valid_read:
            fallback_variants = (("original", self.preprocess_original(prepared)),)
            for variant_name, variant_image in fallback_variants:
                if variant_image is None:
                    continue
                result = read_variant(variant_name, variant_image)
                if result is not None:
                    candidates.append(result)
                    if (
                        result["is_indonesia_pattern"]
                        and result["confidence"] >= 0.85
                    ):
                        break

        if not candidates:
            empty["elapsed"] = time.perf_counter() - total_start
            empty["paddle_inference_elapsed"] = paddle_inference_elapsed
            empty["variants_tried"] = attempted_variants
            save_debug_crop_pair(
                image, prepared, "", 0.0, False, context=context
            )
            return empty

        # Pilih hasil terbaik
        best = max(candidates, key=self._score_result)

        # Bonus konsistensi lintas variant
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
        best["paddle_inference_elapsed"] = paddle_inference_elapsed
        best["variants_tried"] = attempted_variants
        best["formatted"] = self._format_plate(best.get("text", ""))
        best["text"] = self.normalize_text(best.get("text", ""))
        best["is_indonesia_pattern"] = self.is_valid_indonesian_plate(best["text"])
        best["region_code"] = self._extract_region_code(best["text"])

        # Simpan sampel perbandingan crop mentah vs prep untuk evaluasi
        save_debug_crop_pair(
            image,
            prepared,
            best.get("text", ""),
            best.get("confidence", 0.0),
            best.get("is_indonesia_pattern", False),
            context=context,
        )

        if self.verbose:
            print(
                f"[OCR BEST] {best['formatted']} | "
                f"raw={best.get('raw_text', '-')} | "
                f"conf={best['confidence']:.3f} | "
                f"variant={best['variant']} | "
                f"votes={best.get('cross_variant_votes', 1)} | "
                f"region={best.get('region_code')} | "
                f"time={best['elapsed']:.2f}s"
            )

        return best