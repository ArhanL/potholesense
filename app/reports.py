"""Council-facing report generation (CSV + PDF).

UK councils generally accept pothole reports containing: location (lat/lon and
where possible a road name), date observed, a severity indication and a photo.
This module produces both a machine-readable CSV (for bulk submission or FOI
style data sharing) and a human-readable PDF dossier with evidence images.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path

from config import EVIDENCE_DIR
from app import storage
from app.severity import priority_rank

CSV_COLUMNS = [
    "reference", "road_name", "latitude", "longitude", "severity",
    "width_m", "length_m",
    "priority", "first_observed", "last_observed", "times_observed",
    "detector_confidence", "evidence_file", "google_maps_link",
]


def _rows(potholes: list[dict]) -> list[dict]:
    rows = []
    for p in potholes:
        rows.append({
            "reference": f"PS-{p['id']:05d}",
            "road_name": p.get("road_name") or "",
            "latitude": round(p["lat"], 6),
            "longitude": round(p["lon"], 6),
            "severity": p["severity"],
            "width_m": round(p["width_m"], 2) if p.get("width_m") else "",
            "length_m": round(p["length_m"], 2) if p.get("length_m") else "",
            "priority": round(priority_rank(p["severity"], p["sightings"],
                                            p["max_conf"]), 1),
            "first_observed": p["first_seen"],
            "last_observed": p["last_seen"],
            "times_observed": p["sightings"],
            "detector_confidence": round(p["max_conf"], 3),
            "evidence_file": p["evidence"] or "",
            "google_maps_link": f"https://www.google.com/maps?q={p['lat']:.6f},{p['lon']:.6f}",
        })
    rows.sort(key=lambda r: -r["priority"])
    return rows


def to_csv(potholes: list[dict] | None = None) -> str:
    potholes = potholes if potholes is not None else storage.all_potholes()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    w.writerows(_rows(potholes))
    return buf.getvalue()


def to_pdf(path: Path, potholes: list[dict] | None = None,
           council: str = "Local Highways Authority",
           reporter: str = "PotholeSense automated survey") -> Path:
    """Write a PDF dossier. Requires reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image as RLImage, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    potholes = potholes if potholes is not None else storage.all_potholes()
    rows = _rows(potholes)
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    story = []
    story.append(Paragraph("Road Surface Defect Report", h1))
    story.append(Paragraph(
        f"Submitted to: <b>{council}</b><br/>"
        f"Source: {reporter}<br/>"
        f"Generated: {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')}<br/>"
        f"Defects included: <b>{len(rows)}</b>", body))
    story.append(Spacer(1, 6 * mm))

    counts = {s: sum(1 for r in rows if r["severity"] == s)
              for s in ("high", "medium", "low", "unknown")}
    story.append(Paragraph(
        f"Severity breakdown &mdash; high: <b>{counts['high']}</b>, "
        f"medium: <b>{counts['medium']}</b>, low: <b>{counts['low']}</b>, "
        f"unclassified: <b>{counts['unknown']}</b>.", body))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Severity is banded on the defect's width measured on the road plane, "
        "not on its apparent size in the image: <b>low</b> below 300&nbsp;mm, "
        "<b>medium</b> 300&ndash;600&nbsp;mm, <b>high</b> above 600&nbsp;mm. "
        "300&nbsp;mm is the surface width most commonly cited by UK highway "
        "authorities as an intervention level, alongside a 40&nbsp;mm depth "
        "criterion. <b>Depth is not measured</b> &mdash; it cannot be recovered "
        "from a single camera &mdash; so these figures are advisory and do not "
        "replace physical inspection. Defects recorded too far from the camera "
        "to be measured reliably are listed as unclassified rather than "
        "estimated.", small))
    story.append(Spacer(1, 6 * mm))

    table_data = [["Ref", "Location", "Latitude", "Longitude", "Severity",
                   "Width", "Seen"]]
    for r in rows:
        table_data.append([
            r["reference"],
            Paragraph(r["road_name"] or "&mdash;", small),
            f"{r['latitude']:.5f}", f"{r['longitude']:.5f}",
            r["severity"].upper(),
            f"{r['width_m']:.2f} m" if r["width_m"] != "" else "—",
            str(r["times_observed"]),
        ])
    t = Table(table_data, repeatRows=1, hAlign="LEFT",
              colWidths=[20 * mm, 38 * mm, 23 * mm, 23 * mm, 22 * mm,
                         18 * mm, 12 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2933")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b8c1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f7")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)

    # Evidence pages
    with_evidence = [r for r in rows if r["evidence_file"]]
    if with_evidence:
        story.append(PageBreak())
        story.append(Paragraph("Photographic Evidence", styles["Heading2"]))
        for r in with_evidence:
            img_path = EVIDENCE_DIR / Path(r["evidence_file"]).name
            if not img_path.exists():
                continue
            story.append(Spacer(1, 4 * mm))
            where = f" &mdash; {r['road_name']}" if r["road_name"] else ""
            size = f" &mdash; {r['width_m']:.2f} m wide" if r["width_m"] != "" else ""
            story.append(Paragraph(
                f"<b>{r['reference']}</b> &mdash; {r['severity'].upper()}{where}"
                f"{size}<br/>{r['latitude']:.5f}, {r['longitude']:.5f} "
                f"(<link href='{r['google_maps_link']}'>map</link>)", body))
            try:
                story.append(RLImage(str(img_path), width=120 * mm, height=90 * mm,
                                     kind="proportional"))
            except Exception:
                story.append(Paragraph("[evidence image unreadable]", small))

    doc.build(story)
    return path
