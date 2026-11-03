"""
WSGI Entry Point untuk Wiko Vision (Production Deployment)
Mendukung:
- Ubuntu Linux (Gunicorn): gunicorn --workers 1 --threads 12 --worker-class gthread -b 0.0.0.0:5000 wsgi:app
- Windows / Standalone: python wsgi.py (otomatis memakai Waitress multi-threaded jika terinstall)
"""
import os
import sys

# Pastikan camera worker dan background OCR aktif saat dijalankan via WSGI server
os.environ["WERKZEUG_RUN_MAIN"] = "true"

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    try:
        from waitress import serve
        print(f"[WSGI] Menjalankan server produksi Waitress di http://{host}:{port} (threads=12)")
        serve(app, host=host, port=port, threads=12)
    except ImportError:
        print(f"[WSGI] Waitress tidak ditemukan. Menjalankan fallback dev server di http://{host}:{port}")
        app.run(host=host, port=port, debug=False, threaded=True)
