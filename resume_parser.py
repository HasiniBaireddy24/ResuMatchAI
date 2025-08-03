import pdfplumber
import os

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using the pdfplumber library.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    
    return text

if __name__ == '__main__':
    test_pdf_path = os.path.join(os.path.dirname(__file__), 'sample_resume.pdf')
    if os.path.exists(test_pdf_path):
        extracted_text = extract_text_from_pdf(test_pdf_path)
        print("--- Extracted Text from PDF ---")
        print(extracted_text)
    else:
        print(f"File not found: {test_pdf_path}")
