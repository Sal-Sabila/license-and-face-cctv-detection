"""
Wrapper untuk menjaga kompatibilitas impor lama `from plat_ocr import PlateOCR`.
Mengarahkan ke implementasi terbaru berkinerja tinggi di `ai.plate.ocr.PlateOCR`.
"""
from ai.plate.ocr import PlateOCR

__all__ = ["PlateOCR"]