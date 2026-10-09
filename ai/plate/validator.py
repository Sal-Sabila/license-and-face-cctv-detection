"""
Validator & koreksi plat nomor Indonesia.

Modul ini mengoreksi hasil OCR mentah supaya sesuai format plat Indonesia:
  - [1-2 HURUF kode wilayah] [1-4 ANGKA] [0-3 HURUF seri]

Contoh:
    "IS1500IF" -> (tidak valid, IS bukan kode wilayah) -> coba koreksi
    "I5001P"   -> "1500IP"? atau "I5001P"? -> pilih yang valid
    "E057SR"   -> "E 57 SR" (hapus leading zero)
"""

import re

# KODE WILAYAH RESMI INDONESIA
# Sumber: Peraturan Kapolri & data Samsat
# Format: 1-2 huruf
VALID_REGION_CODES = {
    # Sumatera
    "A", "AA", "AB", "AD", "AE", "AG", "BA", "BB", "BD", "BE", "BG", "BH",
    "BK", "BL", "BM", "BN", "BP", "BR", "BT",
    # DKI Jakarta & sekitarnya
    "B", "D", "E", "F", "T",
    # Jawa Barat
    "D", "E", "F", "T", "Z", "AA", "AB", "AD",
    # Jawa Tengah & DIY
    "H", "K", "R", "AA", "AB", "AD",
    # Jawa Timur
    "L", "M", "N", "P", "S", "W", "AG",
    # Kalimantan
    "DA", "KB", "KH", "KT", "KU",
    # Sulawesi
    "DD", "DN", "DP", "DT", "DW", "DM", "DL", "DB",
    # Bali & Nusa Tenggara
    "DK", "ED", "EA", "EB", "DH",
    # Maluku & Papua
    "DE", "DG", "PA", "PB", "PD",
}

# Kode wilayah 1 huruf yang valid (subset dari VALID_REGION_CODES)
VALID_SINGLE_LETTER_REGIONS = {c for c in VALID_REGION_CODES if len(c) == 1}
VALID_DOUBLE_LETTER_REGIONS = {c for c in VALID_REGION_CODES if len(c) == 2}


# KOREKSI KARAKTER OCR YANG UMUM
# Huruf yang sering salah baca menjadi angka, dan sebaliknya
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "5": "S",
    "6": "G", "8": "B", "9": "g",
}
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2",
    "S": "5",
    "G": "6", "b": "6",
    "B": "8",
    "g": "9", "q": "9",
}


def _correct_letter_to_digit(ch: str) -> str:
    return LETTER_TO_DIGIT.get(ch.upper(), ch)


def _correct_digit_to_letter(ch: str) -> str:
    return DIGIT_TO_LETTER.get(ch, ch)


# NORMALISASI DASAR
def normalize_plate_text(text: str) -> str:
    """Hilangkan semua karakter non-alphanumeric, uppercase."""
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper().strip())


# PARSING: PISAHKAN PREFIX / ANGKA / SUFFIX
_PATTERN = re.compile(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})$")


def parse_plate(text: str):
    """
    Parse plat menjadi (prefix, numbers, suffix).
    Return None kalau tidak match pola dasar.
    """
    text = normalize_plate_text(text)
    m = _PATTERN.match(text)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


# KOREKSI PREFIX (KODE WILAYAH)
def _try_correct_prefix(prefix: str) -> list:
    """
    Coba koreksi prefix supaya jadi kode wilayah valid.
    Return list kandidat, urut dari paling mungkin.
    """
    candidates = []

    # Kalau prefix sudah valid, prioritaskan
    if prefix in VALID_REGION_CODES:
        return [prefix]

    # Koreksi karakter digit -> huruf
    corrected = "".join(_correct_digit_to_letter(c) for c in prefix)

    if corrected in VALID_REGION_CODES:
        candidates.append(corrected)

    # Kalau prefix 1 huruf dan valid, prioritaskan
    if len(prefix) == 1 and prefix in VALID_SINGLE_LETTER_REGIONS:
        candidates.append(prefix)

    # Kalau prefix 2 huruf:
    if len(prefix) == 2:
        # Coba 1 huruf pertama saja
        if prefix[0] in VALID_SINGLE_LETTER_REGIONS:
            candidates.append(prefix[0])
        # Coba 1 huruf kedua saja
        if prefix[1] in VALID_SINGLE_LETTER_REGIONS:
            candidates.append(prefix[1])
        # Coba kombinasi koreksi
        if corrected != prefix and corrected in VALID_REGION_CODES:
            candidates.append(corrected)

    # Deduplicate, pertahankan urutan
    seen = set()
    result = []
    for c in candidates:
        if c not in seen and c in VALID_REGION_CODES:
            seen.add(c)
            result.append(c)
    return result


# KOREKSI ANGKA
def _try_correct_numbers(numbers: str) -> list:
    """
    Coba koreksi angka. Angka tidak boleh leading zero (>1 digit).
    """
    candidates = []
    corrected = "".join(_correct_letter_to_digit(c) for c in numbers)

    if corrected.isdigit():
        candidates.append(corrected)

    if numbers.isdigit():
        candidates.append(numbers)

    # Hapus leading zero
    cleaned = []
    for c in candidates:
        stripped = c.lstrip("0") or "0"
        if 1 <= len(stripped) <= 4 and stripped not in cleaned:
            cleaned.append(stripped)
    return cleaned


# KOREKSI SUFFIX (SERI)
def _try_correct_suffix(suffix: str) -> list:
    """
    Coba koreksi suffix. Suffix harus huruf murni.
    """
    if not suffix:
        return [""]
    candidates = []
    corrected = "".join(_correct_digit_to_letter(c) for c in suffix)
    if corrected.isalpha():
        candidates.append(corrected)
    if suffix.isalpha():
        candidates.append(suffix)
    # Dedup
    seen = set()
    result = []
    for c in candidates:
        if c not in seen and (not c or c.isalpha()):
            seen.add(c)
            result.append(c)
    return result or [suffix]


# MAIN: VALIDASI & KOREKSI
def correct_and_validate_plate(raw_text: str, ocr_confidence: float = 0.0):
    """
    Koreksi & validasi plat nomor Indonesia.

    Return dict:
        {
            "original": str,
            "corrected": str,        # hasil koreksi terbaik
            "valid": bool,           # apakah format valid
            "confidence_penalty": float,  # 0.0 = tidak ada koreksi, makin besar = makin banyak koreksi
            "candidates": list[str],  # semua kandidat yang mungkin
        }
    """
    original = normalize_plate_text(raw_text)

    if not original or len(original) < 3:
        return {
            "original": original,
            "corrected": original,
            "valid": False,
            "confidence_penalty": 0.0,
            "candidates": [],
        }

    # Coba parse as-is
    parsed = parse_plate(original)

    # Kalau tidak match pola, coba koreksi total
    if parsed is None:
        # Coba koreksi: huruf di awal, angka di tengah, huruf di akhir
        m = re.match(r"^([A-Z0-9]{1,2})([A-Z0-9]{1,4})([A-Z0-9]{0,3})$", original)
        if not m:
            return {
                "original": original,
                "corrected": original,
                "valid": False,
                "confidence_penalty": 0.0,
                "candidates": [],
            }
        raw_prefix, raw_numbers, raw_suffix = m.groups()
    else:
        raw_prefix, raw_numbers, raw_suffix = parsed

    # Buat semua kombinasi kandidat
    prefix_candidates = _try_correct_prefix(raw_prefix)
    number_candidates = _try_correct_numbers(raw_numbers)
    suffix_candidates = _try_correct_suffix(raw_suffix)

    # Kalau tidak ada prefix valid, berarti tidak bisa dikoreksi
    if not prefix_candidates:
        return {
            "original": original,
            "corrected": original,
            "valid": False,
            "confidence_penalty": 0.0,
            "candidates": [],
        }

    # Bangun kandidat final
    all_candidates = []
    for p in prefix_candidates:
        for n in number_candidates:
            for s in suffix_candidates:
                candidate = f"{p}{n}{s}"
                # Validasi format
                if _PATTERN.match(candidate):
                    # Hitung penalty (jumlah karakter yang dikoreksi)
                    penalty = _compute_penalty(original, candidate)
                    all_candidates.append((candidate, penalty))

    if not all_candidates:
        return {
            "original": original,
            "corrected": original,
            "valid": False,
            "confidence_penalty": 0.0,
            "candidates": [],
        }

    # Sort: penalty kecil dulu, lalu panjang lebih baik
    all_candidates.sort(key=lambda x: (x[1], -len(x[0])))
    best, best_penalty = all_candidates[0]

    return {
        "original": original,
        "corrected": best,
        "valid": True,
        "confidence_penalty": best_penalty / max(len(original), 1),
        "candidates": [c for c, _ in all_candidates[:5]],
    }


def _compute_penalty(original: str, corrected: str) -> int:
    """Hitung berapa karakter yang diubah dari original ke corrected."""
    if len(original) != len(corrected):
        return abs(len(original) - len(corrected)) + 2
    return sum(1 for a, b in zip(original, corrected) if a != b)


def is_valid_plate(text: str) -> bool:
    """Cek cepat apakah text adalah plat Indonesia valid."""
    result = correct_and_validate_plate(text)
    return result["valid"]


# FORMAT UNTUK DISPLAY
def format_plate_for_display(text: str) -> str:
    """
    Format plat untuk display: "AB1234CD" -> "AB 1234 CD"
    """
    text = normalize_plate_text(text)
    m = _PATTERN.match(text)
    if not m:
        return text
    prefix, numbers, suffix = m.groups()
    parts = [prefix, numbers]
    if suffix:
        parts.append(suffix)
    return " ".join(parts)