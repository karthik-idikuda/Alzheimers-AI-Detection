
from fpdf import FPDF
import os
import sys

# Handing encoding errors by defining a class that replaces non-latin-1 chars
def sanitize(text):
    replacements = {
        '\u2018': "'", '\u2019': "'", # Smart quotes
        '\u201C': '"', '\u201D': '"', # Smart double quotes
        '\u2013': '-', '\u2014': '-', # Dashes
        '\u2026': '...',             # Ellipsis
        '\u00A0': ' ',               # Non-breaking space
        '\u2192': '->',              # Right arrow
        '\u03C6': 'phi',             # Greek phi
        '\u00D7': 'x',               # Multiplication sign
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Fallback: remove any other non-latin-1 characters
    return text.encode('latin-1', 'replace').decode('latin-1')

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'NeuroXAI: Advanced Project Report', 0, 1, 'C')
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, num, label):
        # Arial 12
        self.set_font('Arial', 'B', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        label = sanitize(label)
        self.cell(0, 6, '%s' % (label), 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, body):
        # Times 12
        self.set_font('Times', '', 12)
        # Output justified text
        self.multi_cell(0, 5, body)
        # Line break
        self.ln()

def generate_pdf():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Read the markdown file
    with open('NeuroXAI_Technical_Report.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(2)
            continue
            
        line = sanitize(line)
        
        if line.startswith('### '):
            pdf.chapter_title(0, line.replace('### ', ''))
        elif line.startswith('**') and line.endswith('**'):
             pdf.set_font('Times', 'B', 12)
             pdf.multi_cell(0, 5, line.replace('**', ''))
             pdf.set_font('Times', '', 12)
        elif line.startswith('* '):
            pdf.cell(5) # Indent
            pdf.multi_cell(0, 5, chr(149) + ' ' + line[2:])
        else:
            pdf.multi_cell(0, 5, line)

    pdf.output('NeuroXAI_Advanced_Technical_Report.pdf', 'F')
    print("PDF generated successfully: NeuroXAI_Advanced_Technical_Report.pdf")

if __name__ == '__main__':
    generate_pdf()
