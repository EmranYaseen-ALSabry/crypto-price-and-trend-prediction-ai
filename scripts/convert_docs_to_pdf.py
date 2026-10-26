#!/usr/bin/env python3
"""Converts docs/Crypto_Price_Prediction_AI_Documentation.html to PDF using Edge or Chrome."""

import subprocess
import shutil
from pathlib import Path


def main() -> None:
    html_path = Path("docs/Crypto_Price_Prediction_AI_Documentation.html").resolve()
    pdf_docs_path = Path("docs/Crypto_Price_Prediction_AI_Documentation.pdf").resolve()
    pdf_root_path = Path("Crypto_Price_Prediction_AI_Documentation.pdf").resolve()

    if not html_path.exists():
        print(f"[!] Error: {html_path} does not exist.")
        return

    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    browser_bin = None
    for p in edge_paths + chrome_paths:
        if Path(p).exists():
            browser_bin = p
            break

    if not browser_bin:
        print("[!] Neither Edge nor Chrome found on system.")
        return

    print(f"[+] Using browser: {browser_bin}")
    print(f"[+] Input HTML:   {html_path}")
    print(f"[+] Output PDF:  {pdf_docs_path}")

    cmd = [
        browser_bin,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_docs_path}",
        str(html_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if pdf_docs_path.exists() and pdf_docs_path.stat().st_size > 1000:
        size_kb = pdf_docs_path.stat().st_size / 1024
        print(f"[OK] Successfully generated PDF: {pdf_docs_path} ({size_kb:.1f} KB)")
        shutil.copy2(pdf_docs_path, pdf_root_path)
        print(f"[OK] Successfully copied to root: {pdf_root_path}")
    else:
        print(f"[!] Failed to generate PDF. Exit code: {res.returncode}, Error: {res.stderr}")


if __name__ == "__main__":
    main()
