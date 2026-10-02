import hmac
import os
import secrets
import time
from urllib.parse import urlparse

from flask import Blueprint, redirect, render_template, request, session, url_for


auth_bp = Blueprint("auth", __name__)


def _safe_next_url(target):
    if not target:
        return url_for("dashboard_page.dashboard")
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc or not target.startswith("/") or target.startswith("//"):
        return url_for("dashboard_page.dashboard")
    return target


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("authenticated"):
        return redirect(url_for("dashboard_page.dashboard"))

    error = None
    email = os.environ.get("APP_LOGIN_EMAIL", "").strip()
    password = os.environ.get("APP_LOGIN_PASSWORD", "")

    if request.method == "POST":
        submitted_email = request.form.get("email", "").strip()
        submitted_password = request.form.get("password", "")
        configured = bool(email and password)
        email_matches = configured and hmac.compare_digest(
            submitted_email.casefold(), email.casefold()
        )
        password_matches = configured and hmac.compare_digest(
            submitted_password, password
        )

        if email_matches and password_matches:
            session.clear()
            session["authenticated"] = True
            session["user_email"] = email
            session["logout_token"] = secrets.token_urlsafe(32)
            session["last_activity"] = time.time()
            return redirect(_safe_next_url(request.form.get("next")))
        error = "Login belum dikonfigurasi." if not configured else "Email atau password salah."

    return render_template(
        "login.html",
        error=error,
        email=request.form.get("email", ""),
        next_url=_safe_next_url(request.args.get("next")),
    )


@auth_bp.route("/logout", methods=["POST"])
def logout():
    token = request.form.get("logout_token", "")
    if not session.get("logout_token") or not hmac.compare_digest(
        token, session["logout_token"]
    ):
        return "Invalid logout request", 400
    session.clear()
    return redirect(url_for("auth.login"))