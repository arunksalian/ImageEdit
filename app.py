import os
from flask import Flask, request, render_template, flash, send_from_directory, url_for, send_file, make_response
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
import tempfile
from PyPDF2 import PdfReader, PdfWriter
import io
import logging
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    # Get file extension
    ext = original_filename.rsplit('.', 1)[1].lower()
    # Generate unique name using timestamp and UUID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}.{ext}"

def preprocess_image(image):
    """Preprocess image for better OCR results"""
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply different preprocessing techniques
    processed_images = []
    
    # Original grayscale
    processed_images.append(gray)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_images.append(thresh)
    
    # Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    processed_images.append(denoised)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    processed_images.append(sharpened)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    processed_images.append(contrast)
    
    # Convert back to PIL Images
    return [Image.fromarray(img) for img in processed_images]

def extract_text_with_ocr(image, config):
    """Extract text using OCR with different preprocessing"""
    # Get preprocessed images
    processed_images = preprocess_image(image)
    
    # Try OCR on each preprocessed image with different configurations
    texts = []
    configs = [
        config,  # Original config
        config + " -c textord_heavy_nr=1 -c textord_min_linesize=2.5",  # Enhanced config
        config + " -c textord_force_make_prop_words=1",  # Proportional words config
        config + " -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1"  # Combined config
    ]
    
    for img in processed_images:
        for cfg in configs:
            try:
                # Try different PSM modes
                for psm in [6, 4, 3, 1]:
                    current_config = cfg.replace("--psm 6", f"--psm {psm}")
                    text = pytesseract.image_to_string(img, config=current_config)
                    if text.strip():
                        texts.append(text)
            except Exception as e:
                logger.error(f"OCR error: {str(e)}")
    
    # Combine all texts
    return " ".join(texts)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    logger.info(f"Serving uploaded file: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/redact-pdf', methods=['POST'])
def redact_pdf():
    try:
        data = request.get_json()
        redacted_words = data.get('redacted_words', [])
        logger.info(f"Received redaction request for {len(redacted_words)} words")
        logger.debug(f"Redacted words: {redacted_words}")
        
        # Get the current PDF file from the session or request
        current_pdf = request.cookies.get('current_pdf')
        if not current_pdf:
            logger.error("No PDF file selected")
            return 'No PDF file selected', 400
            
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], current_pdf)
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return 'PDF file not found', 404
            
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Open the PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        logger.info(f"PDF opened successfully. Number of pages: {len(doc)}")
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.info(f"Processing page {page_num + 1}")
            
            # Get the page dimensions
            page_rect = page.rect
            logger.debug(f"Page dimensions: {page_rect}")
            
            # Create redaction rectangles for selected words
            for word_info in redacted_words:
                word_text = word_info['text']
                line_num = int(word_info['line'])
                logger.debug(f"Processing word: '{word_text}' from line {line_num}")
                
                # Search for the word on the page (case-insensitive)
                word_instances = page.search_for(word_text, case_sensitive=False)
                logger.info(f"Found {len(word_instances)} instances of word '{word_text}' on page {page_num + 1}")
                
                # If no instances found, try with OCR
                if len(word_instances) == 0:
                    # Convert page to image with higher resolution
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Use OCR with custom configuration
                    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                    ocr_text = pytesseract.image_to_string(img, config=custom_config)
                    logger.debug(f"OCR text: {ocr_text}")
                    
                    # Get word boxes from OCR
                    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=custom_config)
                    
                    # Search for word in OCR text and boxes
                    for i, text in enumerate(data['text']):
                        if word_text.lower() in text.lower():
                            x = data['left'][i]
                            y = data['top'][i]
                            w = data['width'][i]
                            h = data['height'][i]
                            # Scale coordinates back to original size
                            rect = fitz.Rect(x/2, y/2, (x+w)/2, (y+h)/2)
                            word_instances.append(rect)
                            logger.info(f"Found word '{word_text}' using OCR at coordinates: {rect}")
                
                # Redact each instance of the word
                for rect in word_instances:
                    # Add padding to the rectangle
                    padding = 2
                    rect.x0 -= padding
                    rect.y0 -= padding
                    rect.x1 += padding
                    rect.y1 += padding
                    
                    logger.debug(f"Creating redaction rectangle at coordinates: {rect}")
                    
                    # Create redaction annotation
                    try:
                        annot = page.add_redact_annot(rect)
                        annot.set_appearance(fill=(0, 0, 0))  # Black fill
                        annot.update()
                        logger.debug(f"Successfully added redaction for word '{word_text}' at coordinates: {rect}")
                    except Exception as e:
                        logger.error(f"Error creating redaction annotation: {str(e)}")
                        continue
            
            # Apply redactions
            try:
                page.apply_redactions()
                logger.info(f"Successfully applied redactions on page {page_num + 1}")
            except Exception as e:
                logger.error(f"Error applying redactions on page {page_num + 1}: {str(e)}")
        
        # Save to a temporary file
        temp_dir = tempfile.gettempdir()
        redacted_filename = f"redacted_{current_pdf}"
        redacted_path = os.path.join(temp_dir, redacted_filename)
        
        logger.info(f"Saving redacted PDF to: {redacted_path}")
        
        # Save the redacted PDF
        try:
            doc.save(redacted_path, garbage=4, deflate=True, clean=True)
            doc.close()
            logger.info("Successfully saved redacted PDF")
            
            # Verify the file was created and has content
            if os.path.exists(redacted_path):
                file_size = os.path.getsize(redacted_path)
                logger.info(f"Redacted PDF file size: {file_size} bytes")
                if file_size == 0:
                    raise Exception("Generated PDF file is empty")
            else:
                raise Exception("Failed to create redacted PDF file")
            
        except Exception as e:
            logger.error(f"Error saving redacted PDF: {str(e)}")
            return str(e), 500
        
        # Send the file
        logger.info("Sending redacted PDF to client")
        return send_file(
            redacted_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='redacted_document.pdf'
        )
        
    except Exception as e:
        logger.error(f"Error in redact_pdf: {str(e)}", exc_info=True)
        return str(e), 500

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        logger.info("Received file upload request")
        
        if 'file' not in request.files:
            logger.warning("No file part in request")
            flash('No file part')
            return render_template('index.html')
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            flash('No selected file')
            return render_template('index.html')
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = generate_unique_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            logger.info(f"Saving uploaded file to: {filepath}")
            
            # Save the file
            file.save(filepath)
            
            # Extract text from PDF
            try:
                # First try to extract text directly from PDF
                doc = fitz.open(filepath)
                content = []
                total_pages = len(doc)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    logger.info(f"Processing page {page_num + 1}")
                    
                    # Try multiple text extraction methods
                    # 1. Direct text extraction
                    text = page.get_text()
                    logger.debug(f"Direct text extraction result: {text[:200]}...")
                    
                    # 2. Try getting text with blocks
                    blocks = page.get_text("blocks")
                    block_text = ""
                    for block in blocks:
                        if block[6] == 0:  # Text block
                            block_text += block[4] + " "
                    logger.debug(f"Block text extraction result: {block_text[:200]}...")
                    
                    # 3. Try getting text with words
                    words = page.get_text("words")
                    word_text = " ".join([w[4] for w in words])
                    logger.debug(f"Word text extraction result: {word_text[:200]}...")
                    
                    # 4. Try getting text with rawdict
                    rawdict = page.get_text("rawdict")
                    raw_text = ""
                    for block in rawdict.get("blocks", []):
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                raw_text += span.get("text", "") + " "
                    logger.debug(f"Raw text extraction result: {raw_text[:200]}...")
                    
                    # Combine all extraction methods
                    combined_text = text + " " + block_text + " " + word_text + " " + raw_text
                    
                    # If no text found or text is too short, use OCR
                    if not combined_text.strip() or len(combined_text.strip()) < 10:
                        logger.info(f"Using OCR for page {page_num + 1}")
                        # Convert page to image with higher resolution
                        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))  # Increased resolution
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # Use OCR with custom configuration for better word detection
                        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1'
                        ocr_text = extract_text_with_ocr(img, custom_config)
                        logger.debug(f"OCR text result: {ocr_text[:200]}...")
                        
                        combined_text += " " + ocr_text
                    
                    # Clean up the text
                    combined_text = combined_text.replace('\n\n', '\n').strip()
                    
                    # Split text into words and clean them
                    words = combined_text.split()
                    cleaned_words = []
                    for word in words:
                        # Remove special characters but keep letters, numbers, and common symbols
                        cleaned_word = ''.join(c for c in word if c.isalnum() or c in '-_')
                        if cleaned_word:  # Only add non-empty words
                            # Keep both original and uppercase versions for words that might be in uppercase
                            cleaned_words.append(cleaned_word)
                            if cleaned_word.isupper():
                                cleaned_words.append(cleaned_word.lower())
                            elif cleaned_word.islower():
                                cleaned_words.append(cleaned_word.upper())
                    
                    # Remove duplicates while preserving case variations
                    seen = set()
                    cleaned_words = [x for x in cleaned_words if not (x.lower() in seen or seen.add(x.lower()))]
                    
                    # Sort words alphabetically (case-insensitive)
                    cleaned_words.sort(key=str.lower)
                    
                    # Join words back with spaces
                    cleaned_text = ' '.join(cleaned_words)
                    
                    # Log all found words for debugging
                    logger.info(f"All words found on page {page_num + 1} (sorted): {cleaned_words}")
                    
                    # Add page marker and cleaned text
                    content.append({
                        'page_num': page_num + 1,
                        'text': cleaned_text,
                        'words': cleaned_words
                    })
                
                # Generate URL for the uploaded file
                pdf_url = url_for('uploaded_file', filename=filename)
                
                logger.info("Successfully processed PDF")
                
                # Create response with template
                response = make_response(render_template('index.html', 
                    content=content, 
                    pdf_url=pdf_url,
                    total_pages=total_pages))
                # Set cookie with current PDF filename
                response.set_cookie('current_pdf', filename)
                
                # Close the document after we're done with it
                doc.close()
                
                return response
            
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
                flash(f'Error processing PDF: {str(e)}')
                # Clean up the file if there was an error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return render_template('index.html')
        
        logger.warning(f"Invalid file type: {file.filename}")
        flash('Invalid file type. Please upload a PDF file.')
        return render_template('index.html')
    
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("Starting PDF Text Extractor application")
    app.run(host='0.0.0.0', port=5000, debug=True) 