import PyPDF2
import sys
import pytesseract
from pdf2image import convert_from_path
import os
from PIL import Image
import subprocess

def find_tesseract_data():
    """Find Tesseract data directory by searching for eng.traineddata"""
    try:
        # Try to find eng.traineddata using find command
        result = subprocess.run(['find', '/usr', '-name', 'eng.traineddata'], 
                              capture_output=True, text=True)
        if result.stdout:
            return os.path.dirname(result.stdout.split('\n')[0])
    except Exception:
        pass
    
    # Fallback paths
    possible_paths = [
        os.getenv('TESSDATA_PREFIX'),
        '/usr/share/tesseract-ocr/4.00/tessdata',
        '/usr/share/tesseract-ocr/tessdata',
        r'C:\Program Files\Tesseract-OCR\tessdata'
    ]
    
    for path in possible_paths:
        if path and os.path.exists(os.path.join(path, 'eng.traineddata')):
            return path
    
    return None

# Set Tesseract path from environment variable or use default
tesseract_cmd = os.getenv('TESSERACT_CMD', r'C:\Program Files\Tesseract-OCR\tesseract.exe')

# Find and set Tesseract data directory
tessdata_prefix = find_tesseract_data()
if tessdata_prefix:
    print(f"Found Tesseract data at: {tessdata_prefix}")
    os.environ['TESSDATA_PREFIX'] = tessdata_prefix
else:
    print("Warning: Could not find Tesseract data directory")

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def verify_tesseract():
    """Verify Tesseract installation and configuration"""
    try:
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        print(f"Tesseract data path: {os.getenv('TESSDATA_PREFIX', 'Not set')}")
        
        # Try a simple OCR test
        test_image = Image.new('RGB', (100, 100), color='white')
        pytesseract.image_to_string(test_image, lang='eng')
        return True
    except Exception as e:
        print(f"Error verifying Tesseract: {str(e)}")
        return False

def read_pdf(file_path):
    try:
        # Verify Tesseract is working
        if not verify_tesseract():
            raise Exception("Tesseract OCR is not properly configured")

        # First try to read as regular PDF
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            content = [f"Total number of pages: {num_pages}"]
            
            # Try to extract text from each page
            has_text = False
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():  # If page has text
                    has_text = True
                    content.append(f"\n--- Page {page_num + 1} ---")
                    content.append(text)
            
            # If no text was found, the PDF is likely scanned
            if not has_text:
                content = [f"Total number of pages: {num_pages}"]
                # Convert PDF to images
                images = convert_from_path(file_path)
                
                # Process each page with OCR
                for i, image in enumerate(images):
                    # Convert image to text using OCR
                    text = pytesseract.image_to_string(image, lang='eng')
                    
                    content.append(f"\n--- Page {i + 1} ---")
                    content.append(text)
            
            return content
                
    except FileNotFoundError:
        raise Exception(f"The file '{file_path}' was not found.")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf_reader.py <path_to_pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    content = read_pdf(pdf_path)
    for line in content:
        print(line)

if __name__ == "__main__":
    main() 