import unittest
import os
import tempfile
import shutil
from app import app, allowed_file, generate_unique_filename
import fitz  # PyMuPDF
from io import BytesIO

class TestPDFRedaction(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        app.config['UPLOAD_FOLDER'] = self.test_dir
        
        # Use the specific PDF file for testing
        self.sample_pdf_path = r"C:\Users\CITPL\Documents\CIC\doc.pdf"
        if not os.path.exists(self.sample_pdf_path):
            raise FileNotFoundError(f"Test PDF file not found at: {self.sample_pdf_path}")
        
        # Create a copy of the PDF in the test directory
        self.test_pdf_path = os.path.join(self.test_dir, 'test.pdf')
        shutil.copy2(self.sample_pdf_path, self.test_pdf_path)
        
    def tearDown(self):
        """Clean up after each test"""
        shutil.rmtree(self.test_dir)
    
    def test_allowed_file(self):
        """Test file extension validation"""
        self.assertTrue(allowed_file('test.pdf'))
        self.assertFalse(allowed_file('test.txt'))
        self.assertFalse(allowed_file('test'))
        self.assertFalse(allowed_file(''))
    
    def test_generate_unique_filename(self):
        """Test unique filename generation"""
        filename1 = generate_unique_filename('test.pdf')
        filename2 = generate_unique_filename('test.pdf')
        self.assertNotEqual(filename1, filename2)
        self.assertTrue(filename1.endswith('.pdf'))
    
    def test_upload_pdf(self):
        """Test PDF upload functionality"""
        with open(self.test_pdf_path, 'rb') as f:
            response = self.app.post(
                '/',
                data={'file': (f, 'test.pdf')},
                content_type='multipart/form-data'
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'PDF Reader', response.data)
    
    def test_redact_specific_word(self):
        """Test redaction of the word 'REDRAWN'"""
        # First upload the PDF
        with open(self.test_pdf_path, 'rb') as f:
            response = self.app.post(
                '/',
                data={'file': (f, 'test.pdf')},
                content_type='multipart/form-data'
            )
        
        # Get the cookie with the PDF filename
        cookies = response.headers.getlist('Set-Cookie')
        pdf_cookie = next((c for c in cookies if 'current_pdf' in c), None)
        self.assertIsNotNone(pdf_cookie)
        
        # Extract the filename from the cookie
        import re
        filename_match = re.search(r'current_pdf=([^;]+)', pdf_cookie)
        self.assertIsNotNone(filename_match)
        filename = filename_match.group(1)
        
        # Test redaction with specific word "REDRAWN"
        redaction_data = {
            'redacted_words': [
                {'text': 'REDRAWN', 'line': '0'}
            ]
        }
        
        response = self.app.post(
            '/redact-pdf',
            json=redaction_data,
            headers={'Cookie': f'current_pdf={filename}'}
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')
        
        # Verify the redacted PDF
        redacted_pdf = BytesIO(response.data)
        doc = fitz.open(stream=redacted_pdf, filetype="pdf")
        page = doc[0]
        redacted_text = page.get_text()
        
        # Check if the word "REDRAWN" is not in the text
        self.assertNotIn('REDRAWN', redacted_text)
        doc.close()
    
    def test_redact_multiple_words(self):
        """Test redaction of multiple words"""
        # First upload the PDF
        with open(self.test_pdf_path, 'rb') as f:
            response = self.app.post(
                '/',
                data={'file': (f, 'test.pdf')},
                content_type='multipart/form-data'
            )
        
        # Get the cookie with the PDF filename
        cookies = response.headers.getlist('Set-Cookie')
        pdf_cookie = next((c for c in cookies if 'current_pdf' in c), None)
        filename = re.search(r'current_pdf=([^;]+)', pdf_cookie).group(1)
        
        # Test redaction with multiple words
        redaction_data = {
            'redacted_words': [
                {'text': 'REDRAWN', 'line': '0'},
                {'text': 'DRAWN', 'line': '0'},
                {'text': 'TO', 'line': '0'}
            ]
        }
        
        response = self.app.post(
            '/redact-pdf',
            json=redaction_data,
            headers={'Cookie': f'current_pdf={filename}'}
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')
        
        # Verify the redacted PDF
        redacted_pdf = BytesIO(response.data)
        doc = fitz.open(stream=redacted_pdf, filetype="pdf")
        page = doc[0]
        redacted_text = page.get_text()
        
        # Check if all redacted words are not in the text
        for word in ['REDRAWN', 'DRAWN', 'TO']:
            self.assertNotIn(word, redacted_text)
        doc.close()
    
    def test_invalid_file_upload(self):
        """Test uploading invalid file types"""
        # Create a text file
        text_file_path = os.path.join(self.test_dir, 'test.txt')
        with open(text_file_path, 'w') as f:
            f.write('This is a text file')
        
        with open(text_file_path, 'rb') as f:
            response = self.app.post(
                '/',
                data={'file': (f, 'test.txt')},
                content_type='multipart/form-data'
            )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid file type', response.data)
    
    def test_redact_nonexistent_pdf(self):
        """Test redaction with non-existent PDF"""
        response = self.app.post(
            '/redact-pdf',
            json={'redacted_words': [{'text': 'REDRAWN', 'line': '0'}]},
            headers={'Cookie': 'current_pdf=nonexistent.pdf'}
        )
        
        self.assertEqual(response.status_code, 404)
    
    def test_redact_no_words(self):
        """Test redaction with no words selected"""
        # First upload the PDF
        with open(self.test_pdf_path, 'rb') as f:
            response = self.app.post(
                '/',
                data={'file': (f, 'test.pdf')},
                content_type='multipart/form-data'
            )
        
        # Get the cookie with the PDF filename
        cookies = response.headers.getlist('Set-Cookie')
        pdf_cookie = next((c for c in cookies if 'current_pdf' in c), None)
        filename = re.search(r'current_pdf=([^;]+)', pdf_cookie).group(1)
        
        # Test redaction with empty word list
        response = self.app.post(
            '/redact-pdf',
            json={'redacted_words': []},
            headers={'Cookie': f'current_pdf={filename}'}
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')

if __name__ == '__main__':
    unittest.main() 