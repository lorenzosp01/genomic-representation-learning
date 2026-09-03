#!/usr/bin/env python3
"""Generate a well-formatted PDF from the contrastive learning report markdown."""

import markdown
from weasyprint import HTML
from pathlib import Path
import re

REPORT_DIR = Path(__file__).parent
MD_FILE = REPORT_DIR / "REPORT_CONTRASTIVE_PAPERFAITHFUL.md"
PDF_FILE = REPORT_DIR / "REPORT_CONTRASTIVE_PAPERFAITHFUL.pdf"

CSS = """
@page {
    size: A4;
    margin: 2cm 1.5cm;
    @bottom-center {
        content: "Pagina " counter(page) " di " counter(pages);
        font-size: 9px;
        color: #666;
    }
}

body {
    font-family: 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif;
    font-size: 11px;
    line-height: 1.5;
    color: #1a1a1a;
}

h1 {
    font-size: 22px;
    color: #1a5276;
    border-bottom: 3px solid #2980b9;
    padding-bottom: 8px;
    margin-top: 0;
    page-break-after: avoid;
}

h2 {
    font-size: 17px;
    color: #1a5276;
    border-bottom: 1.5px solid #85c1e9;
    padding-bottom: 5px;
    margin-top: 28px;
    page-break-after: avoid;
}

h3 {
    font-size: 14px;
    color: #2471a3;
    margin-top: 20px;
    page-break-after: avoid;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 10px;
    page-break-inside: avoid;
}

th {
    background-color: #2980b9;
    color: white;
    font-weight: bold;
    padding: 7px 10px;
    text-align: left;
    border: 1px solid #1a6da0;
}

td {
    padding: 5px 10px;
    border: 1px solid #bdc3c7;
    text-align: left;
}

tr:nth-child(even) { background-color: #eaf2f8; }
tr:nth-child(odd)  { background-color: #ffffff; }

td strong, th strong { color: #1a5276; }

blockquote {
    border-left: 4px solid #2980b9;
    background-color: #eaf2f8;
    padding: 10px 15px;
    margin: 12px 0;
    font-style: italic;
    color: #2c3e50;
    page-break-inside: avoid;
}

blockquote strong { font-style: normal; }

code {
    background-color: #ecf0f1;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace;
    font-size: 10px;
    color: #c0392b;
}

pre {
    background-color: #2c3e50;
    color: #ecf0f1;
    padding: 12px 15px;
    border-radius: 5px;
    font-size: 9px;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    color: #ecf0f1;
    padding: 0;
}

hr {
    border: none;
    border-top: 1px solid #bdc3c7;
    margin: 20px 0;
}

ol, ul { margin: 8px 0; padding-left: 25px; }
li { margin-bottom: 4px; }

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 15px auto;
    page-break-inside: avoid;
}
"""


def main():
    md_text = MD_FILE.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    # Resolve image paths to absolute file:// URIs
    def resolve_img(m):
        src = m.group(1)
        abs_path = (REPORT_DIR / src).resolve()
        if abs_path.exists():
            return f'src="file://{abs_path}"'
        return m.group(0)

    html_body = re.sub(r'src="([^"]+)"', resolve_img, html_body)

    full_html = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    print("Generating PDF...")
    HTML(string=full_html, base_url=str(REPORT_DIR)).write_pdf(str(PDF_FILE))
    size_kb = PDF_FILE.stat().st_size / 1024
    print(f"✅ PDF generato: {PDF_FILE} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
