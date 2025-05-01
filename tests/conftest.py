import pytest
import os
import tempfile
import shutil
from app import app

@pytest.fixture
def client():
    """Create a test client for the app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def test_dir():
    """Create a temporary directory for test files"""
    test_dir = tempfile.mkdtemp()
    app.config['UPLOAD_FOLDER'] = test_dir
    yield test_dir
    shutil.rmtree(test_dir)

@pytest.fixture
def sample_pdf_path():
    """Return the path to the sample PDF file"""
    pdf_path = r"C:\Users\CITPL\Documents\CIC\doc.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Sample PDF file not found at: {pdf_path}")
    return pdf_path

@pytest.fixture
def test_pdf_path(test_dir, sample_pdf_path):
    """Create a copy of the sample PDF in the test directory"""
    test_pdf_path = os.path.join(test_dir, 'test.pdf')
    shutil.copy2(sample_pdf_path, test_pdf_path)
    return test_pdf_path 