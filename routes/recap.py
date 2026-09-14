from flask import Blueprint, render_template


recap_bp = Blueprint("recap_page", __name__)


@recap_bp.route("/recap")
def recap():
    return render_template(
        "index.html",
        page="recap",
        heading="Rekapitulasi",
        subtitle="Agregasi aktivitas kendaraan, plat, dan orang dari seluruh CCTV."
    )
