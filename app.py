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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
from flask_restx import Api, Resource, fields
import shutil

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

# Global configuration
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp'
TESSERACT_TEMP_DIR = '/tmp/tesseract'
REDACTED_FOLDER = 'redacted'  # New folder for storing redacted files

def ensure_tesseract_temp_dir():
    """Ensure Tesseract temporary directory exists and is accessible"""
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(TESSERACT_TEMP_DIR):
            os.makedirs(TESSERACT_TEMP_DIR, mode=0o777, exist_ok=True)
            logger.info(f"Created Tesseract temp directory: {TESSERACT_TEMP_DIR}")
        
        # Try to set permissions
        try:
            os.chmod(TESSERACT_TEMP_DIR, 0o777)
            logger.info(f"Set permissions for {TESSERACT_TEMP_DIR}")
        except PermissionError:
            logger.warning(f"Could not change permissions for {TESSERACT_TEMP_DIR} - this is normal for mounted volumes")
        
        # Test if directory is writable
        test_file = os.path.join(TESSERACT_TEMP_DIR, 'test.txt')
        try:
            # First try to write
            with open(test_file, 'w') as f:
                f.write('test')
            logger.info(f"Successfully wrote test file: {test_file}")
            
            # Then try to read
            with open(test_file, 'r') as f:
                content = f.read()
            logger.info(f"Successfully read test file: {test_file}")
            
            # Finally try to remove
            if os.path.exists(test_file):
                os.remove(test_file)
                logger.info(f"Successfully removed test file: {test_file}")
            
        except Exception as e:
            logger.error(f"Error testing Tesseract temp directory: {str(e)}")
            # Don't raise the error, just log it
            # The application will handle any subsequent file operation errors
            
    except Exception as e:
        logger.error(f"Error setting up Tesseract temp directory: {str(e)}")
        # Don't raise the error, just log it
        # The application will handle any subsequent file operation errors

def init_app():
    """Initialize the Flask application with proper error handling"""
    try:
        app = Flask(__name__)
        app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

        # Initialize Flask-RESTX
        api = Api(app, 
            version='1.0',
            title='PDF Text Extractor API',
            description='A RESTful API for PDF text extraction and redaction',
            doc='/swagger'
        )

        # Create namespaces
        ns = api.namespace('pdf', description='PDF operations')

        # Create necessary directories with proper permissions
        for directory in [UPLOAD_FOLDER, TEMP_FOLDER, REDACTED_FOLDER]:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory, mode=0o777, exist_ok=True)
                try:
                    os.chmod(directory, 0o777)
                except PermissionError:
                    logger.warning(f"Could not change permissions for {directory} - this is normal for mounted volumes")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {str(e)}")
                raise

        # Ensure Tesseract temp directory exists and is writable
        ensure_tesseract_temp_dir()

        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        app.config['TEMP_FOLDER'] = TEMP_FOLDER
        app.config['TESSERACT_TEMP_DIR'] = TESSERACT_TEMP_DIR
        app.config['REDACTED_FOLDER'] = REDACTED_FOLDER

        # Configure Tesseract
        tesseract_cmd = os.getenv('TESSERACT_CMD', '/usr/bin/tesseract')
        tesseract_data_prefix = os.getenv('TESSDATA_PREFIX', '/usr/share/tesseract-ocr/4.00/tessdata')

        if not os.path.exists(tesseract_cmd):
            logger.error(f"Tesseract executable not found at {tesseract_cmd}")
            raise RuntimeError(f"Tesseract executable not found at {tesseract_cmd}")

        if not os.path.exists(tesseract_data_prefix):
            logger.error(f"Tesseract data directory not found at {tesseract_data_prefix}")
            raise RuntimeError(f"Tesseract data directory not found at {tesseract_data_prefix}")

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        os.environ['TESSDATA_PREFIX'] = tesseract_data_prefix
        os.environ['TMPDIR'] = TESSERACT_TEMP_DIR

        # Verify Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Error verifying Tesseract installation: {str(e)}")
            raise

        return app, api, ns

    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        raise

# Initialize the application
app, api, ns = init_app()

# Define models for Swagger documentation
upload_model = api.model('Upload', {
    'file': fields.Raw(description='PDF file to upload')
})

redact_model = api.model('Redact', {
    'redacted_words': fields.List(fields.String, description='List of words to redact')
})

download_model = api.model('Download', {
    'filename': fields.String(description='Name of the redacted file to download')
})

# Update the combined model definition
upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type='FileStorage', required=True, help='PDF file to upload and redact')
upload_parser.add_argument('words', location='form', type='str', required=True, action='append', help='Words to redact')

# Add new parser for word extraction
word_extract_parser = api.parser()
word_extract_parser.add_argument('file', location='files', type='FileStorage', required=True, help='PDF file to extract words from')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=4)

# Cache for OCR results
ocr_cache = {}
ocr_cache_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    # Get file extension
    ext = original_filename.rsplit('.', 1)[1].lower()
    # Generate unique name using timestamp and UUID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}.{ext}"

@lru_cache(maxsize=100)
def preprocess_image(image_data):
    """Preprocess image for better OCR results with caching"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply different preprocessing techniques
    processed_images = []
    
    # Original grayscale
    processed_images.append(gray)
    
    # Adaptive thresholding with different block sizes
    for block_size in [11, 15, 21]:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)
        processed_images.append(thresh)
    
    # Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)
    
    # Denoise with different parameters
    for h in [10, 15, 20]:
        denoised = cv2.fastNlMeansDenoising(gray, None, h, 7, 21)
        processed_images.append(denoised)
    
    # Sharpen with different kernels
    kernels = [
        np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]),  # Standard sharpening
        np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]),      # Light sharpening
        np.array([[-2,-2,-2], [-2,17,-2], [-2,-2,-2]])  # Strong sharpening
    ]
    for kernel in kernels:
        sharpened = cv2.filter2D(gray, -1, kernel)
        processed_images.append(sharpened)
    
    # Increase contrast with different parameters
    for clip_limit in [2.0, 3.0, 4.0]:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        processed_images.append(contrast)
    
    # Convert back to PIL Images
    return [Image.fromarray(img) for img in processed_images]

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        if os.path.exists(TESSERACT_TEMP_DIR):
            # Remove all files in the directory but keep the directory itself
            for filename in os.listdir(TESSERACT_TEMP_DIR):
                file_path = os.path.join(TESSERACT_TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting temporary file {file_path}: {str(e)}")
            
            # Ensure directory has proper permissions
            try:
                os.chmod(TESSERACT_TEMP_DIR, 0o777)
            except PermissionError:
                logger.warning(f"Could not change permissions for {TESSERACT_TEMP_DIR} - this is normal for mounted volumes")
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {str(e)}")
        # Try to recreate the directory if it was accidentally removed
        try:
            if not os.path.exists(TESSERACT_TEMP_DIR):
                os.makedirs(TESSERACT_TEMP_DIR, mode=0o777, exist_ok=True)
        except Exception as e2:
            logger.error(f"Error recreating temp directory: {str(e2)}")

def extract_text_with_ocr(image, config):
    """Extract text using OCR with different preprocessing and caching"""
    try:
        # Ensure Tesseract temp directory exists and is writable
        ensure_tesseract_temp_dir()
        logger.debug("Tesseract temp directory verified")

        # Convert image to bytes for caching
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        logger.debug("Image converted to bytes for caching")
        
        # Check cache first
        with ocr_cache_lock:
            if img_byte_arr in ocr_cache:
                logger.debug("Using cached OCR result")
                return ocr_cache[img_byte_arr]
        
        # Get preprocessed images
        processed_images = preprocess_image(img_byte_arr)
        logger.debug(f"Generated {len(processed_images)} preprocessed images")
        
        # Try OCR on each preprocessed image with different configurations
        texts = []
        configs = [
            # Base configurations
            config,
            # Enhanced configurations for better word detection
            config + " --oem 3 --psm 6 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1",
            config + " --oem 3 --psm 4 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1",
            config + " --oem 3 --psm 3 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1",
            # Configurations with different character sets
            config + " --oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            config + " --oem 3 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            # Configurations with different dictionaries
            config + " --oem 3 --psm 6 -c load_system_dawg=1 -c load_freq_dawg=1",
            config + " --oem 3 --psm 4 -c load_system_dawg=1 -c load_freq_dawg=1",
            # Configurations with different PSM modes
            config + " --oem 3 --psm 1",  # Automatic page segmentation with OSD
            config + " --oem 3 --psm 0",  # Orientation and script detection
            # Combined best settings
            config + " --oem 3 --psm 6 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1 -c load_system_dawg=1 -c load_freq_dawg=1"
        ]
        
        for img_idx, img in enumerate(processed_images):
            logger.debug(f"Processing preprocessed image {img_idx + 1}")
            for cfg_idx, cfg in enumerate(configs):
                logger.debug(f"Trying OCR configuration {cfg_idx + 1}")
                try:
                    # Create a temporary file for the image
                    temp_file = None
                    try:
                        # Ensure temp directory exists before creating file
                        ensure_tesseract_temp_dir()
                        
                        temp_file = tempfile.NamedTemporaryFile(suffix='.png', dir=TESSERACT_TEMP_DIR, delete=False)
                        img.save(temp_file.name)
                        temp_file.close()
                        logger.debug(f"Created temporary file: {temp_file.name}")
                        
                        # Get both text and word boxes
                        text = pytesseract.image_to_string(temp_file.name, config=cfg)
                        if text.strip():
                            logger.debug(f"Found text with config {cfg_idx + 1}: {text[:100]}...")
                            texts.append(text)
                            
                            # Also get word boxes for better word detection
                            data = pytesseract.image_to_data(temp_file.name, output_type=pytesseract.Output.DICT, config=cfg)
                            words = []
                            for i, word in enumerate(data['text']):
                                if word.strip():
                                    words.append(word.strip())
                            if words:
                                logger.debug(f"Found words with config {cfg_idx + 1}: {', '.join(words)}")
                                texts.append(' '.join(words))
                        
                    finally:
                        # Clean up the temporary file
                        if temp_file and os.path.exists(temp_file.name):
                            try:
                                os.unlink(temp_file.name)
                                logger.debug(f"Cleaned up temporary file: {temp_file.name}")
                            except Exception as e:
                                logger.warning(f"Error deleting temporary file {temp_file.name}: {str(e)}")
                        
                except Exception as e:
                    logger.error(f"OCR error with config {cfg_idx + 1}: {str(e)}")
                    continue
        
        # Combine all texts and clean up
        result = " ".join(texts)
        
        # Clean up the text
        result = ' '.join(result.split())  # Remove extra whitespace
        result = result.replace('\n', ' ')  # Replace newlines with spaces
        
        logger.debug(f"Final OCR result: {result[:100]}...")
        
        # Cache the result
        with ocr_cache_lock:
            ocr_cache[img_byte_arr] = result
        
        return result
        
    except Exception as e:
        logger.error(f"Error in extract_text_with_ocr: {str(e)}")
        logger.exception("Full traceback:")  # Add full traceback
        return ""

def process_page(page, page_num, redact_words=None):
    """Process a single page in parallel"""
    try:
        logger.info(f"Processing page {page_num + 1}")
        
        # If redaction is requested, process the page
        if redact_words:
            logger.info(f"Processing redaction for words: {redact_words}")
            
            # Convert page to image for OCR with higher resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(8, 8))  # Increased resolution for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Get word boxes from OCR with multiple configurations
            configs = [
                # Standard configurations
                r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1',
                r'--oem 3 --psm 4 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1',
                r'--oem 3 --psm 3 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1',
                # Configurations with different character sets
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                r'--oem 3 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
                # Configurations with different dictionaries
                r'--oem 3 --psm 6 -c load_system_dawg=1 -c load_freq_dawg=1',
                r'--oem 3 --psm 4 -c load_system_dawg=1 -c load_freq_dawg=1',
                # Configurations with different PSM modes
                r'--oem 3 --psm 1',  # Automatic page segmentation with OSD
                r'--oem 3 --psm 0',  # Orientation and script detection
                # Combined best settings
                r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_force_make_prop_words=1 -c load_system_dawg=1 -c load_freq_dawg=1'
            ]
            
            # Process each word to redact
            for word_to_redact in redact_words:
                logger.debug(f"Looking for word to redact: {word_to_redact}")
                found_rects = []
                
                # Try each OCR configuration
                for config in configs:
                    try:
                        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
                        
                        # Search for the word in OCR results
                        for i, text in enumerate(data['text']):
                            if text.strip():
                                # Try multiple matching strategies
                                text_lower = text.lower()
                                word_lower = word_to_redact.lower()
                                
                                if (word_lower in text_lower or  # Partial match
                                    text_lower == word_lower or  # Exact match
                                    word_to_redact.upper() in text.upper() or  # Uppercase match
                                    word_to_redact in text):  # Direct match
                                    
                                    x = data['left'][i]
                                    y = data['top'][i]
                                    w = data['width'][i]
                                    h = data['height'][i]
                                    conf = data['conf'][i]
                                    
                                    # Only process if confidence is above threshold
                                    if conf > 30:  # Adjust confidence threshold as needed
                                        # Scale coordinates back to original size
                                        rect = fitz.Rect(x/8, y/8, (x+w)/8, (y+h)/8)
                                        # Add padding to the rectangle
                                        padding = 4  # Increased padding
                                        rect.x0 -= padding
                                        rect.y0 -= padding
                                        rect.x1 += padding
                                        rect.y1 += padding
                                        found_rects.append(rect)
                                        logger.debug(f"Found word '{word_to_redact}' at coordinates: {rect} with confidence: {conf}")
                    except Exception as e:
                        logger.error(f"Error in OCR config {config}: {str(e)}")
                        continue
                
                # Also try direct text search with PyMuPDF
                try:
                    # Try different search variations
                    search_terms = [
                        word_to_redact,
                        word_to_redact.upper(),
                        word_to_redact.lower(),
                        word_to_redact.capitalize()
                    ]
                    
                    for term in search_terms:
                        word_instances = page.search_for(term, case_sensitive=False)
                        for rect in word_instances:
                            # Add padding to the rectangle
                            padding = 4  # Increased padding
                            rect.x0 -= padding
                            rect.y0 -= padding
                            rect.x1 += padding
                            rect.y1 += padding
                            found_rects.append(rect)
                            logger.debug(f"Found word '{term}' via direct search at coordinates: {rect}")
                except Exception as e:
                    logger.error(f"Error in direct text search: {str(e)}")
                
                # Create redaction annotations for all found instances
                for rect in found_rects:
                    try:
                        # Create a slightly larger rectangle for better coverage
                        expanded_rect = fitz.Rect(
                            rect.x0 - 1,
                            rect.y0 - 1,
                            rect.x1 + 1,
                            rect.y1 + 1
                        )
                        
                        # Create redaction annotation
                        annot = page.add_redact_annot(expanded_rect)
                        
                        # Set the redaction appearance using the correct method
                        annot.set_info(title="Redaction", content="Redacted text")
                        annot.set_border(width=1)
                        annot.set_colors(stroke=(0, 0, 0), fill=(0, 0, 0))
                        annot.update()
                        
                        # Draw a black rectangle to ensure complete coverage
                        page.draw_rect(expanded_rect, color=(0, 0, 0), fill=(0, 0, 0))
                        
                        logger.debug(f"Added black redaction for word '{word_to_redact}' at {expanded_rect}")
                    except Exception as e:
                        logger.error(f"Error creating redaction annotation: {str(e)}")
                        continue
            
            # Apply redactions
            try:
                # First apply the redactions
                page.apply_redactions()
                
                # Then draw black rectangles over the redacted areas to ensure complete coverage
                for rect in found_rects:
                    page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
                
                logger.info(f"Successfully applied redactions on page {page_num + 1}")
            except Exception as e:
                logger.error(f"Error applying redactions on page {page_num + 1}: {str(e)}")
        
        # Extract text for response
        text = page.get_text()
        words = text.split()
        cleaned_words = []
        for word in words:
            # Remove special characters but keep letters, numbers, and common symbols
            cleaned_word = ''.join(c for c in word if c.isalnum() or c in '-_')
            if cleaned_word:  # Only add non-empty words
                cleaned_words.append(cleaned_word)
        
        # Remove duplicates while preserving case variations
        seen = set()
        cleaned_words = [x for x in cleaned_words if not (x.lower() in seen or seen.add(x.lower()))]
        
        # Sort words alphabetically (case-insensitive)
        cleaned_words.sort(key=str.lower)
        
        return {
            'page_num': page_num + 1,
            'text': ' '.join(cleaned_words),
            'words': cleaned_words
        }
    except Exception as e:
        logger.error(f"Error processing page {page_num + 1}: {str(e)}")
        logger.exception("Full traceback:")
        return None

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    logger.info(f"Serving uploaded file: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@ns.route('/upload')
class PDFUpload(Resource):
    @ns.expect(upload_model)
    @ns.doc('upload_pdf',
        responses={
            200: 'Success',
            400: 'Invalid file type',
            500: 'Server error'
        })
    def post(self):
        """Upload a PDF file for processing"""
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No selected file'}, 400
        
        if file and allowed_file(file.filename):
            filename = generate_unique_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                doc = fitz.open(filepath)
                total_pages = len(doc)
                
                futures = []
                for page_num in range(total_pages):
                    page = doc[page_num]
                    futures.append(thread_pool.submit(process_page, page, page_num))
                
                content = []
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        content.append(result)
                
                content.sort(key=lambda x: x['page_num'])
                pdf_url = url_for('uploaded_file', filename=filename)
                
                doc.close()
                
                return {
                    'message': 'File uploaded successfully',
                    'filename': filename,
                    'total_pages': total_pages,
                    'content': content,
                    'pdf_url': pdf_url
                }
            
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return {'error': str(e)}, 500
        
        return {'error': 'Invalid file type'}, 400

@ns.route('/redact')
class PDFRedact(Resource):
    @ns.expect(redact_model)
    @ns.doc('redact_pdf',
        responses={
            200: 'Success',
            400: 'No PDF selected',
            404: 'PDF not found',
            500: 'Server error'
        })
    def post(self):
        """Redact words from the current PDF"""
        try:
            data = request.get_json()
            redacted_words = data.get('redacted_words', [])
            logger.info(f"Received redaction request for words: {redacted_words}")
            
            current_pdf = request.cookies.get('current_pdf')
            if not current_pdf:
                return {'error': 'No PDF file selected'}, 400
            
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], current_pdf)
            if not os.path.exists(pdf_path):
                return {'error': 'PDF file not found'}, 404
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            logger.info(f"PDF opened successfully. Number of pages: {len(doc)}")
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                logger.info(f"Processing page {page_num + 1}")
                process_page(page, page_num, redacted_words)
            
            # Generate unique filename for redacted file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            redacted_filename = f"redacted_{timestamp}_{unique_id}.pdf"
            redacted_path = os.path.join(app.config['REDACTED_FOLDER'], redacted_filename)
            
            logger.info(f"Saving redacted PDF to: {redacted_path}")
            
            # Save the redacted PDF with optimized settings
            try:
                doc.save(redacted_path, 
                        garbage=4,  # Maximum garbage collection
                        deflate=True,  # Compress streams
                        clean=True,  # Clean redundant elements
                        linear=True,  # Optimize for web viewing
                        pretty=False,  # Minimize file size
                        ascii=False)  # Use binary encoding
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
                
                return {
                    'message': 'PDF redacted successfully',
                    'filename': redacted_filename,
                    'download_url': url_for('pdf_pdf_download', filename=redacted_filename, _external=True)
                }
                
            except Exception as e:
                logger.error(f"Error saving redacted PDF: {str(e)}")
                return {'error': str(e)}, 500
            
        except Exception as e:
            logger.error(f"Error in redact_pdf: {str(e)}")
            return {'error': str(e)}, 500

@ns.route('/download/<filename>')
class PDFDownload(Resource):
    @ns.doc('download_pdf',
        responses={
            200: 'Success - Returns redacted PDF file',
            404: 'File not found',
            500: 'Server error'
        })
    def get(self, filename):
        """Download a redacted PDF file"""
        try:
            # Validate filename
            if not filename.startswith('redacted_'):
                return {'error': 'Invalid filename'}, 400
                
            file_path = os.path.join(app.config['REDACTED_FOLDER'], filename)
            if not os.path.exists(file_path):
                return {'error': 'File not found'}, 404
                
            logger.info(f"Sending redacted PDF file: {filename}")
            return send_file(
                file_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
            
        except Exception as e:
            logger.error(f"Error downloading file {filename}: {str(e)}")
            return {'error': str(e)}, 500

@ns.route('/upload-and-redact')
class PDFUploadAndRedact(Resource):
    @ns.expect(upload_parser)
    @ns.doc('upload_and_redact_pdf',
        responses={
            200: 'Success - Returns redacted PDF',
            400: 'Invalid file type or missing parameters',
            500: 'Server error'
        })
    def post(self):
        """Upload a PDF file and redact specified words in a single operation"""
        try:
            # Get file from request.files instead of parser
            if 'file' not in request.files:
                return {'error': 'No file part'}, 400
            
            file = request.files['file']
            if file.filename == '':
                return {'error': 'No selected file'}, 400
            
            # Get words from form data
            words = request.form.getlist('words')
            if not words:
                return {'error': 'No words specified for redaction'}, 400
            
            if file and allowed_file(file.filename):
                # Generate unique filename
                filename = generate_unique_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the file
                file.save(filepath)
                logger.info(f"Saved file to: {filepath}")
                
                try:
                    # Open the PDF with PyMuPDF
                    doc = fitz.open(filepath)
                    logger.info(f"PDF opened successfully. Number of pages: {len(doc)}")
                    
                    # Process each page
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        logger.info(f"Processing page {page_num + 1}")
                        process_page(page, page_num, words)
                    
                    # Generate unique filename for redacted file
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    unique_id = str(uuid.uuid4())[:8]
                    redacted_filename = f"redacted_{timestamp}_{unique_id}.pdf"
                    redacted_path = os.path.join(app.config['REDACTED_FOLDER'], redacted_filename)
                    
                    logger.info(f"Saving redacted PDF to: {redacted_path}")
                    
                    # Save the redacted PDF with optimized settings
                    doc.save(redacted_path, 
                            garbage=4,
                            deflate=True,
                            clean=True,
                            linear=True,
                            pretty=False,
                            ascii=False)
                    doc.close()
                    
                    # Clean up the original file
                    os.remove(filepath)
                    
                    return {
                        'message': 'PDF uploaded and redacted successfully',
                        'filename': redacted_filename,
                        'download_url': url_for('pdf_pdf_download', filename=redacted_filename, _external=True)
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return {'error': str(e)}, 500
            
            return {'error': 'Invalid file type'}, 400
            
        except Exception as e:
            logger.error(f"Error in upload_and_redact: {str(e)}")
            return {'error': str(e)}, 500

# Add cleanup for redacted files
def cleanup_old_files():
    """Clean up old redacted files"""
    try:
        redacted_dir = app.config['REDACTED_FOLDER']
        current_time = datetime.now()
        
        for filename in os.listdir(redacted_dir):
            if filename.startswith('redacted_'):
                file_path = os.path.join(redacted_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                
                # Delete files older than 1 hour
                if (current_time - file_time).total_seconds() > 3600:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
                    except Exception as e:
                        logger.error(f"Error deleting old file {filename}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in cleanup_old_files: {str(e)}")

@app.before_request
def before_request():
    """Clean up temporary files before each request"""
    cleanup_temp_files()
    cleanup_old_files()

@app.after_request
def after_request(response):
    """Clean up temporary files after each request"""
    cleanup_temp_files()
    return response

if __name__ == '__main__':
    logger.info("Starting PDF Text Extractor application")
    app.run(host='0.0.0.0', port=5000, debug=True) 