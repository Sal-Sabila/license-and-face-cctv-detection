import re

# ============================================================
# DAFTAR KODE WILAYAH PLAT NOMOR INDONESIA (TNKB)
# ============================================================
KODE_WILAYAH_INDONESIA = {
    # Sumatra
    "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL", "BM", "BN", "BP",
    # Jabodetabek, Banten, Jabar
    "A", "B", "D", "E", "F", "T", "Z",
    # Jawa Tengah & D.I. Yogyakarta
    "G", "H", "K", "R", "AA", "AB", "AD",
    # Jawa Timur
    "L", "M", "N", "P", "S", "W", "AE", "AG",
    # Bali & Nusa Tenggara
    "DK", "DR", "EA", "DH", "EB", "ED",
    # Kalimantan
    "DA", "KB", "KH", "KT", "KU",
    # Sulawesi
    "DB", "DC", "DD", "DL", "DM", "DN", "DP", "DT",
    # Maluku & Papua
    "DE", "DG", "PA", "PB",
}

# Koreksi kesalahan OCR umum pada 2 huruf kode wilayah (misal AR -> AB, AO -> AD)
PREFIX_CORRECTIONS = {
    "AR": "AB",
    "AP": "AB",
    "AQ": "AB",
    "AO": "AD",
    "AC": "AD",
    "AL": "AB",
}

# ============================================================
# MAPPING AMBIGUITAS KARAKTER (CONFUSION MATRIX)
# ============================================================

# Huruf yang sering salah terbaca padahal seharusnya ANGKA
CHAR_TO_DIGIT = {
    "B": "8",
    "O": "0",
    "D": "0",
    "Q": "0",
    "C": "0",
    "I": "1",
    "L": "1",
    "J": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "A": "4",
    "T": "7",
}

# Angka yang sering salah terbaca padahal seharusnya HURUF
DIGIT_TO_CHAR = {
    "8": "B",
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "9": "P",
}


def to_digits(text: str) -> str:
    """Mengubah karakter yang ambigu menjadi angka."""
    return "".join(CHAR_TO_DIGIT.get(c, c) for c in text.upper())


def to_letters(text: str) -> str:
    """Mengubah angka yang ambigu menjadi huruf."""
    return "".join(DIGIT_TO_CHAR.get(c, c) for c in text.upper())


def is_tax_date_format(text: str) -> bool:
    """
    Mendeteksi apakah potongan teks adalah baris masa berlaku pajak.
    Contoh: '08.28', '11-27', '05 26', '.28', '0828'.
    """
    clean = text.strip()
    if re.search(r"^\d{1,2}[\.\-\/\s]\d{2}$", clean):
        return True
    if re.search(r"^[\.\-\/]\d{2}$", clean):
        return True
    if len(clean) == 4 and clean.isdigit():
        month = int(clean[:2])
        year = int(clean[2:])
        if 1 <= month <= 12 and 20 <= year <= 38:
            return True
    return False


def koreksi_plat_indonesia(raw_text: str) -> dict:
    """
    Menerapkan Post-Processing Rule Engine untuk plat nomor Indonesia:
    Pola standar: [1-2 Huruf Wilayah] [1-4 Angka Nomor] [1-3 Huruf Seri]

    Mengembalikan:
        dict dengan text terformat (spasi), text tanpa spasi, status validitas, dll.
    """
    if not raw_text:
        return {"text": "", "formatted": "", "valid": False, "is_indonesia_pattern": False}

    cleaned = re.sub(r"[^A-Z0-9]", "", str(raw_text).upper().strip())

    if len(cleaned) < 3 or len(cleaned) > 12:
        return {"text": cleaned, "formatted": cleaned, "valid": False, "is_indonesia_pattern": False}

    # ------------------------------------------------------------
    # SKENARIO 1: Teks sudah memiliki pemisah spasi alami
    # Contoh: "AB 1618 SI" atau "B 2750 TRO"
    # ------------------------------------------------------------
    parts = [p for p in re.split(r"[\s\.\-]+", str(raw_text).upper().strip()) if p]
    if len(parts) >= 3:
        raw_p1 = to_letters(parts[0])
        p1 = PREFIX_CORRECTIONS.get(raw_p1, raw_p1)
        p2 = to_digits(parts[1])
        p3 = to_letters("".join(parts[2:]))

        if p1 in KODE_WILAYAH_INDONESIA and p2.isdigit() and p3.isalpha() and len(p3) <= 3:
            formatted = f"{p1} {p2} {p3}"
            return {
                "text": f"{p1}{p2}{p3}",
                "formatted": formatted,
                "valid": True,
                "is_indonesia_pattern": True,
            }

    # Jika 2 bagian: contoh "1480 HK"
    if len(parts) == 2:
        raw_a, raw_b = parts
        if raw_a.isdigit() and len(raw_a) <= 4:
            p_suf = to_letters(raw_b)
            if p_suf.isalpha() and len(p_suf) <= 3:
                return {
                    "text": f"{raw_a}{p_suf}",
                    "formatted": f"{raw_a} {p_suf}",
                    "valid": True,
                    "is_indonesia_pattern": False,
                }

    # ------------------------------------------------------------
    # SKENARIO 2: Teks utuh tanpa pemisah (contoh: "B2750TRO", "AB1618SI", "P4712Y")
    # Coba berbagai kemungkinan pemotongan segmen (Prefix, Number, Suffix)
    # ------------------------------------------------------------
    best_candidate = None

    # Tentukan urutan pengecekan prefix:
    # Jika karakter pertama adalah kode wilayah 1 huruf dan karakter kedua adalah angka,
    # prioritaskan prefix 1 huruf terlebih dahulu agar tidak merusak plat seperti P 4712 Y -> PA 712 Y.
    first_char = cleaned[0] if len(cleaned) > 0 else ""
    second_char = cleaned[1] if len(cleaned) > 1 else ""
    single_codes = {"A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "W", "Z"}

    if first_char in single_codes and (second_char.isdigit() or second_char in CHAR_TO_DIGIT):
        prefix_options = [1, 2]
    else:
        prefix_options = [2, 1]

    for prefix_len in prefix_options:
        if len(cleaned) < prefix_len + 2:
            continue

        raw_p1 = cleaned[:prefix_len]
        p1_candidate = to_letters(raw_p1)
        p1 = PREFIX_CORRECTIONS.get(p1_candidate, p1_candidate)

        is_known_prefix = p1 in KODE_WILAYAH_INDONESIA

        remainder = cleaned[prefix_len:]

        for num_len in range(min(4, len(remainder) - 1), 0, -1):
            raw_p2 = remainder[:num_len]
            raw_p3 = remainder[num_len:]

            p2 = to_digits(raw_p2)
            p3 = to_letters(raw_p3) if raw_p3 else ""

            if not p2.isdigit():
                continue

            if raw_p3 and (not p3.isalpha() or len(p3) > 3):
                continue

            # Hitung mutasi karakter (penalti jika mengubah karakter asli)
            mutations = (
                sum(c1 != c2 for c1, c2 in zip(raw_p1, p1))
                + sum(c1 != c2 for c1, c2 in zip(raw_p2, p2))
                + sum(c1 != c2 for c1, c2 in zip(raw_p3, p3))
            )

            score = 0
            if is_known_prefix:
                score += 15
            elif any(kw[0] == p1[0] for kw in KODE_WILAYAH_INDONESIA if len(kw) == 2):
                score += 5

            # Bonus kecocokan alami (tanpa mutasi)
            if mutations == 0:
                score += 10
            else:
                score -= (mutations * 4)

            # Bonus jika prefix 1 huruf asli tidak dipaksa menjadi 2 huruf
            if prefix_len == 1 and raw_p1 in single_codes:
                score += 5

            if p1.isalpha():
                score += 3
            if p2.isdigit():
                score += 4
            if raw_p2.isdigit():
                score += 3
            if raw_p3 and raw_p3.isalpha():
                score += 3

            # Plat Indonesia umumnya memiliki 1-4 digit angka
            if 1 <= len(p2) <= 4:
                score += 2

            candidate = {
                "text": f"{p1}{p2}{p3}",
                "formatted": f"{p1} {p2} {p3}".strip(),
                "valid": True,
                "is_indonesia_pattern": is_known_prefix,
                "score": score,
            }

            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

    if best_candidate is not None and best_candidate["score"] >= 10:
        return best_candidate

    # Cek format tanpa prefix (contoh "1480HK")
    m_no_prefix = re.match(r"^(\d{1,4})([A-Z]{1,3})$", cleaned)
    if m_no_prefix:
        num, suf = m_no_prefix.groups()
        return {
            "text": f"{num}{suf}",
            "formatted": f"{num} {suf}",
            "valid": True,
            "is_indonesia_pattern": False,
        }

    return {
        "text": cleaned,
        "formatted": cleaned,
        "valid": 3 <= len(cleaned) <= 10,
        "is_indonesia_pattern": False,
    }


def filter_dan_gabung_spasial(rec_texts, rec_boxes, rec_scores, min_confidence=0.20):
    """
    Spatial Filtering untuk PaddleOCR:
    1. Memisahkan baris utama plat vs baris pajak (tanggal berlaku)
    2. Mengurutkan teks pada baris utama dari kiri ke kanan (X)
    3. Menggabungkan segmen dengan format yang tepat
    """
    if not rec_texts:
        return None

    items = []
    has_boxes = rec_boxes is not None and len(rec_boxes) > 0
    has_scores = rec_scores is not None and len(rec_scores) > 0

    for i, raw in enumerate(rec_texts):
        score = float(rec_scores[i]) if has_scores and i < len(rec_scores) else 0.0
        if score < min_confidence:
            continue

        raw_clean = str(raw).strip()
        if not raw_clean:
            continue

        box = None
        if has_boxes and i < len(rec_boxes):
            b = rec_boxes[i]
            if hasattr(b, "tolist"):
                b = b.tolist()
            if len(b) == 4:
                box = b

        if box is not None and len(box) == 4:
            x1, y1, x2, y2 = [float(v) for v in box]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            yc = (y1 + y2) / 2.0
            xc = (x1 + x2) / 2.0
        else:
            x1, y1, x2, y2 = (0.0, 0.0, 0.0, 0.0)
            w, h, yc, xc = (0.0, 0.0, 0.0, 0.0)

        items.append({
            "text": raw_clean,
            "score": score,
            "box": (x1, y1, x2, y2),
            "w": w,
            "h": h,
            "yc": yc,
            "xc": xc,
        })

    if not items:
        return None

    # Jika hanya ada 1 kotak teks, langsung evaluasi
    if len(items) == 1:
        corr = koreksi_plat_indonesia(items[0]["text"])
        return {
            "text": corr["text"] if corr["valid"] else items[0]["text"],
            "formatted": corr["formatted"] if corr["valid"] else items[0]["text"],
            "confidence": items[0]["score"],
            "is_indonesia_pattern": corr["is_indonesia_pattern"],
        }

    # ------------------------------------------------------------
    # IDENTIFIKASI DAN ELIMINASI BARIS PAJAK (TAX DATE)
    # ------------------------------------------------------------
    max_h = max(it["h"] for it in items)
    min_yc = min(it["yc"] for it in items)
    valid_items = []

    for it in items:
        clean_text = re.sub(r"[^A-Z0-9]", "", it["text"].upper())

        # Cek jika teks jelas format tanggal pajak
        if is_tax_date_format(it["text"]):
            continue

        # Jika teks berada di area bawah (yc jauh lebih besar dari min_yc)
        # dan teksnya pendek (hanya angka masa berlaku seperti '12', '28', '190')
        is_bottom = (it["yc"] - min_yc) > max(8.0, max_h * 0.50)
        if is_bottom and (clean_text.isdigit() or len(clean_text) <= 3):
            continue

        # Cek jika tinggi box jauh lebih kecil (<= 60% dari box terbesar) dan posisinya di bawah
        if max_h > 0 and it["h"] <= 0.60 * max_h and is_bottom:
            continue

        valid_items.append(it)

    if not valid_items:
        valid_items = items

    # ------------------------------------------------------------
    # PISAHKAN BERDASARKAN BARIS (HORIZONTAL LINE CLUSTERING)
    # ------------------------------------------------------------
    valid_items.sort(key=lambda it: it["yc"])

    lines = []
    for it in valid_items:
        placed = False
        for line in lines:
            avg_h = sum(x["h"] for x in line) / len(line)
            avg_yc = sum(x["yc"] for x in line) / len(line)
            if abs(it["yc"] - avg_yc) <= max(4.0, avg_h * 0.40):
                line.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])

    # Baris utama plat adalah baris dengan pola plat Indonesia terbaik / score terbesar
    def _line_score(line):
        raw_str = " ".join(x["text"] for x in sorted(line, key=lambda it: it["box"][0]))
        res = koreksi_plat_indonesia(raw_str)
        bonus = 30.0 if res["is_indonesia_pattern"] else (15.0 if res["valid"] else 0.0)
        return bonus + sum(x["score"] for x in line) + sum(x["h"] for x in line)

    best_line = max(lines, key=_line_score)

    # Urutkan elemen pada baris utama dari KIRI ke KANAN (horizontal sort by X)
    best_line.sort(key=lambda it: it["box"][0])

    combined_raw = " ".join(it["text"] for it in best_line)
    avg_conf = sum(it["score"] for it in best_line) / len(best_line)

    corr = koreksi_plat_indonesia(combined_raw)

    return {
        "text": corr["text"] if corr["valid"] else re.sub(r"[^A-Z0-9]", "", combined_raw.upper()),
        "formatted": corr["formatted"] if corr["valid"] else combined_raw,
        "confidence": avg_conf,
        "is_indonesia_pattern": corr["is_indonesia_pattern"],
    }
