#!/usr/bin/env python3
"""
Convert all PDF files in the current directory to .txt files.
Uses pypdf as the primary method; falls back to pdfplumber if needed.
Skips PDFs that already have a corresponding .txt file.
"""

import os
import sys
from pathlib import Path

def convert_pdf_to_text_pypdf(pdf_path, output_path):
    """Extract text from PDF using pypdf library."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  [ERROR] pypdf not installed. Install with: pip install pypdf")
        return False
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  [OK] Converted using pypdf: {pdf_path.name} -> {output_path.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] pypdf failed: {e}")
        return False

def convert_pdf_to_text_pdfplumber(pdf_path, output_path):
    """Extract text from PDF using pdfplumber library."""
    try:
        import pdfplumber
    except ImportError:
        print(f"  [ERROR] pdfplumber not installed. Install with: pip install pdfplumber")
        return False
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page_num, page in enumerate(pdf.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  [OK] Converted using pdfplumber: {pdf_path.name} -> {output_path.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] pdfplumber failed: {e}")
        return False

def main():
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    pdf_files = list(Path('.').glob('*.pdf'))
    
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s). Processing...\n")
    
    for pdf_path in sorted(pdf_files):
        output_path = pdf_path.with_suffix('.txt')
        
        # Skip if .txt already exists
        if output_path.exists():
            print(f"[SKIP] {pdf_path.name} -> {output_path.name} (already exists)")
            continue
        
        print(f"[PROCESSING] {pdf_path.name}")
        
        # Try pypdf first, fall back to pdfplumber
        success = convert_pdf_to_text_pypdf(pdf_path, output_path)
        if not success:
            success = convert_pdf_to_text_pdfplumber(pdf_path, output_path)
        
        if not success:
            print(f"  [FAILED] Could not convert {pdf_path.name}")
        print()
    
    print("Conversion complete!")

if __name__ == '__main__':
    main()
