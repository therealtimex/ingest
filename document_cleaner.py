#!/usr/bin/env python3
"""
Document Cleaner Module
Cleans extracted text from documents by removing headers, footers, artifacts, etc.
"""

import re
from typing import List, Dict, Optional


class DocumentCleaner:
    """
    Intelligent document cleaner that removes common artifacts from extracted text.
    """

    def __init__(
        self,
        remove_headers: bool = True,
        remove_footers: bool = True,
        remove_page_numbers: bool = True,
        remove_empty_pages: bool = True,
        fix_latex: bool = True,
        fix_ligatures: bool = True,
        fix_unicode: bool = True,
        min_page_length: int = 50,
    ):
        """
        Initialize the document cleaner.

        Args:
            remove_headers: Remove repeated headers from pages
            remove_footers: Remove page numbers and footers
            remove_page_numbers: Remove standalone page numbers
            remove_empty_pages: Remove pages with minimal content
            fix_latex: Fix LaTeX artifacts (e.g., "L ATEX" -> "LaTeX")
            fix_ligatures: Fix ligature issues (e.g., "ﬁ" -> "fi")
            fix_unicode: Fix common Unicode issues
            min_page_length: Minimum characters for a page to be kept
        """
        self.remove_headers = remove_headers
        self.remove_footers = remove_footers
        self.remove_page_numbers = remove_page_numbers
        self.remove_empty_pages = remove_empty_pages
        self.fix_latex = fix_latex
        self.fix_ligatures = fix_ligatures
        self.fix_unicode = fix_unicode
        self.min_page_length = min_page_length

        # Common header/footer patterns
        self.header_patterns = [
            r'^Chapter \d+\.',  # "Chapter 1."
            r'^\d+\s+Chapter \d+\.',  # "vi Chapter 1."
            r'^[ivxlcdm]+\s+Chapter',  # Roman numerals + Chapter
        ]

        self.footer_patterns = [
            r'^\d+\s*$',  # Standalone page number
            r'^Page \d+\s*$',  # "Page 1"
            r'^\d+\s+of\s+\d+\s*$',  # "1 of 100"
        ]

    def clean_text(self, text: str) -> str:
        """
        Clean a single text block.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Fix LaTeX artifacts
        if self.fix_latex:
            text = self._fix_latex_artifacts(text)

        # Fix ligatures
        if self.fix_ligatures:
            text = self._fix_ligatures(text)

        # Fix Unicode issues
        if self.fix_unicode:
            text = self._fix_unicode_issues(text)

        # Remove headers/footers from lines
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip headers
            if self.remove_headers and self._is_header(line):
                continue

            # Skip footers
            if self.remove_footers and self._is_footer(line):
                continue

            # Skip standalone page numbers
            if self.remove_page_numbers and self._is_page_number(line):
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def clean_markdown(self, markdown: str) -> str:
        """
        Clean markdown content with page-level awareness.

        Args:
            markdown: Full markdown document

        Returns:
            Cleaned markdown
        """
        # Split into pages
        pages = re.split(r'^## Page \d+\s*$', markdown, flags=re.MULTILINE)

        cleaned_pages = []
        for i, page_content in enumerate(pages):
            if i == 0:  # Header before first page
                if page_content.strip():
                    cleaned_pages.append(page_content.strip())
                continue

            # Clean the page content
            cleaned_content = self.clean_text(page_content)

            # Skip empty pages
            if self.remove_empty_pages:
                if len(cleaned_content) < self.min_page_length:
                    continue

            # Add page marker back
            cleaned_pages.append(f"## Page {i}\n\n{cleaned_content}")

        return '\n\n'.join(cleaned_pages)

    def _is_header(self, line: str) -> bool:
        """Check if a line is a header artifact."""
        for pattern in self.header_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False

    def _is_footer(self, line: str) -> bool:
        """Check if a line is a footer artifact."""
        for pattern in self.footer_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _is_page_number(self, line: str) -> bool:
        """Check if a line is just a page number."""
        return bool(re.match(r'^\d+\s*$', line))

    def _fix_latex_artifacts(self, text: str) -> str:
        """Fix common LaTeX rendering issues."""
        replacements = {
            r'L\s*A\s*TEX': 'LaTeX',
            r'L\s*A\s*T\s*E\s*X': 'LaTeX',
            r'TEX': 'TeX',
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        return text

    def _fix_ligatures(self, text: str) -> str:
        """Fix ligature characters that weren't converted properly."""
        ligatures = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            'ﬅ': 'ft',
            'ﬆ': 'st',
        }

        for ligature, replacement in ligatures.items():
            text = text.replace(ligature, replacement)

        return text

    def _fix_unicode_issues(self, text: str) -> str:
        """Fix common Unicode issues."""
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '--',  # Em dash
            '\u2026': '...',  # Ellipsis
            '\xa0': ' ',  # Non-breaking space
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text

    def detect_repeated_headers(self, pages: List[str]) -> List[str]:
        """
        Detect headers that appear on multiple pages.

        Args:
            pages: List of page contents

        Returns:
            List of repeated header patterns
        """
        # Track first lines of each page
        first_lines = {}
        for page in pages:
            lines = [l for l in page.split('\n') if l.strip()]
            if lines:
                first_line = lines[0]
                first_lines[first_line] = first_lines.get(first_line, 0) + 1

        # Find lines that appear on multiple pages
        repeated = [line for line, count in first_lines.items() if count > 2]
        return repeated

    def clean_with_page_context(self, pages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Clean pages with awareness of the full document context.

        Args:
            pages: List of page dictionaries with 'text' key

        Returns:
            Cleaned pages
        """
        if not pages:
            return pages

        # Detect repeated headers
        page_texts = [p.get('text', '') for p in pages]
        repeated_headers = self.detect_repeated_headers(page_texts)

        cleaned_pages = []
        for page in pages:
            text = page.get('text', '')

            # Remove detected repeated headers
            for header in repeated_headers:
                text = text.replace(header, '', 1)

            # Apply standard cleaning
            cleaned_text = self.clean_text(text)

            # Skip if too short
            if self.remove_empty_pages and len(cleaned_text) < self.min_page_length:
                continue

            # Update page with cleaned text
            cleaned_page = page.copy()
            cleaned_page['text'] = cleaned_text
            cleaned_pages.append(cleaned_page)

        return cleaned_pages


def clean_markdown_file(input_path: str, output_path: str = None, **kwargs) -> str:
    """
    Clean a markdown file and save the result.

    Args:
        input_path: Path to input markdown file
        output_path: Path to save cleaned markdown (optional)
        **kwargs: Arguments to pass to DocumentCleaner

    Returns:
        Path to cleaned file
    """
    from pathlib import Path

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Clean
    cleaner = DocumentCleaner(**kwargs)
    cleaned = cleaner.clean_markdown(content)

    # Determine output path
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    return str(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_cleaner.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    result = clean_markdown_file(input_file, output_file)
    print(f"Cleaned file saved to: {result}")
