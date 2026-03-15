#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def load_quantum_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_pdf(quantum_dir: Path, out_pdf: Path) -> None:
    rows = load_quantum_rows(quantum_dir / "energy_scan.csv")
    quantum_row = next((row for row in rows if row.get("vqe_energy")), None)
    if quantum_row is None:
        raise RuntimeError("No quantum row found in energy_scan.csv")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, leading=15, textColor=colors.HexColor("#17324d")))

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=landscape(letter),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.35 * inch,
    )

    story = []
    story.append(Paragraph("Two-Track Proof Sheet: Kaggle Docking + Local Quantum Demo", styles["Title"]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "This one-page sheet summarizes the two parallel routes currently supporting the project: "
            "(1) the Kaggle docking benchmark used for the main paper claims, and "
            "(2) the local H2 VQE demo used as a compact quantum-method evidence artifact.",
            styles["Small"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    route_a = [
        Paragraph("<b>Route A: Kaggle main benchmark</b>", styles["Section"]),
        Paragraph("Task: pose prediction on Astex targets with FK-SMC + SOCM versus SOCM baseline.", styles["Small"]),
        Paragraph("Current targets: improve SR@2A, lower mmff_fallback_rate and crash_rate, and test whether Claims 1/2/3 remain defensible.", styles["Small"]),
        Paragraph("Claim 1: stability across 3 seeds with lower variance and fewer invalid/fallback events.", styles["Tiny"]),
        Paragraph("Claim 2: better SR@2A under the same time budget, not just under the same step count.", styles["Tiny"]),
        Paragraph("Claim 3: log_Z / rerank signals should correlate with pose quality and improve top-k selection.", styles["Tiny"]),
    ]

    route_b = [
        Paragraph("<b>Route B: local quantum proof-of-concept</b>", styles["Section"]),
        Paragraph("Task: run a minimal VQE experiment on H2 and compare it with a classical force-field baseline.", styles["Small"]),
        Paragraph("Reason: provide direct evidence that the applicant has already implemented a basic quantum molecular simulation workflow, rather than only citing interest in quantum methods.", styles["Small"]),
        Paragraph("Implementation note: PySCF was unavailable on this machine, so the quantum side uses the standard 2-qubit H2 tutorial Hamiltonian and the classical side uses an RDKit UFF bond-length scan.", styles["Tiny"]),
    ]

    columns = [
        [*route_a],
        [*route_b],
    ]
    table = Table(columns, colWidths=[5.0 * inch, 5.0 * inch])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9ab0c3")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d3dde6")),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.12 * inch))

    summary_data = [
        ["Quantum demo output", "Value"],
        ["Bond length used for quantum point", f"{float(quantum_row['bond_length_angstrom']):.3f} A"],
        ["VQE energy", f"{float(quantum_row['vqe_energy']):.8f} Hartree"],
        ["Exact energy", f"{float(quantum_row['exact_energy']):.8f} Hartree"],
        ["|VQE - exact|", f"{float(quantum_row['abs_vqe_minus_exact']):.2e} Hartree"],
        ["Classical reference at same row", f"{float(quantum_row['uff_energy']):.4f} kcal/mol (UFF)"],
    ]
    summary = Table(summary_data, colWidths=[2.8 * inch, 2.1 * inch])
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c7d6")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#edf3f8")]),
            ]
        )
    )

    image_row = []
    for name in ["h2_vqe_curve.png", "h2_uff_curve.png"]:
        image_path = quantum_dir / name
        if image_path.exists():
            image_row.append(Image(str(image_path), width=3.35 * inch, height=2.15 * inch))
    if not image_row:
        raise RuntimeError("Quantum output images are missing.")

    bottom = Table([[summary, image_row[0], image_row[1] if len(image_row) > 1 else ""]], colWidths=[5.0 * inch, 2.35 * inch, 2.35 * inch])
    bottom.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9ab0c3")),
            ]
        )
    )
    story.append(bottom)
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "Interpretation: the Kaggle route is the paper route, where success is judged by docking metrics and claim tables. "
            "The quantum route is the evidence route, where success is judged by a compact, reproducible artifact showing direct hands-on use of VQE and an informed comparison with a classical force field.",
            styles["Tiny"],
        )
    )

    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a one-page two-track project proof PDF.")
    parser.add_argument("--quantum_dir", type=Path, required=True)
    parser.add_argument("--out_pdf", type=Path, required=True)
    args = parser.parse_args()
    build_pdf(args.quantum_dir, args.out_pdf)


if __name__ == "__main__":
    main()
