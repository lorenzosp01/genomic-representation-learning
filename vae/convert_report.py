
import markdown
from weasyprint import HTML, CSS
import os

# Configuration
input_file = "REPORT_ESPERIMENTI.md"
output_file = "REPORT_ESPERIMENTI.pdf"

# Read Markdown
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Convert to HTML
html_content = markdown.markdown(text, extensions=['tables'])

# Add minimal CSS for better styling
css_str = """
    body { font-family: sans-serif; line-height: 1.6; font-size: 12pt; }
    h1, h2, h3 { color: #2c3e50; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
    img { max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ccc; }
    blockquote { border-left: 5px solid #eee; padding-left: 15px; color: #555; }
    code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }
"""

# HTML template with base_url for images
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
</head>
<body>
    {html_content}
</body>
</html>
"""

print(f"Generating PDF: {output_file}...")

# Generate PDF with base_url pointing to current directory for images
HTML(string=full_html, base_url=os.getcwd()).write_pdf(output_file, stylesheets=[CSS(string=css_str)])

print("Done! ✅")
