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
    "reference", "latitude", "longitude", "severity", "priority",
    "first_observed", "last_observed", "times_observed",
    "detector_confidence", "evidence_file", "google_maps_link",
]


def _rows(potholes: list[dict]) -> list[dict]:
    rows = []
    for p in potholes:
        rows.append({
            "reference": f"PS-{p['id']:05d}",
            "latitude": round(p["lat"], 6),
            "longitude": round(p["lon"], 6),
            "severity": p["severity"],
            "priority": round(priority_rank(p["severity"], p["sightings"]), 1),
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
              for s in ("high", "medium", "low")}
    story.append(Paragraph(
        f"Severity breakdown &mdash; high: <b>{counts['high']}</b>, "
        f"medium: <b>{counts['medium']}</b>, low: <b>{counts['low']}</b>.", body))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Severity is an automated estimate derived from the defect's apparent "
        "size in frame and detector confidence, corroborated across repeat "
        "sightings. It is advisory and does not replace physical inspection.",
        small))
    story.append(Spacer(1, 6 * mm))

    table_data = [["Ref", "Latitude", "Longitude", "Severity", "Seen", "Conf."]]
    for r in rows:
        table_data.append([
            r["reference"], f"{r['latitude']:.5f}", f"{r['longitude']:.5f}",
            r["severity"].upper(), str(r["times_observed"]),
            f"{r['detector_confidence']:.2f}",
        ])
    t = Table(table_data, repeatRows=1, hAlign="LEFT",
              colWidths=[22 * mm, 26 * mm, 26 * mm, 24 * mm, 16 * mm, 18 * mm])
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
            story.append(Paragraph(
                f"<b>{r['reference']}</b> &mdash; {r['severity'].upper()} &mdash; "
                f"{r['latitude']:.5f}, {r['longitude']:.5f} "
                f"(<link href='{r['google_maps_link']}'>map</link>)", body))
            try:
                story.append(RLImage(str(img_path), width=120 * mm, height=90 * mm,
                                     kind="proportional"))
            except Exception:
                story.append(Paragraph("[evidence image unreadable]", small))

    doc.build(story)
    return path
