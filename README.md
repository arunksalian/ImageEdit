# PDF Text Extractor

A powerful web application and API for extracting text from both regular and scanned PDF files. This application supports direct text extraction from PDFs and uses OCR (Optical Character Recognition) for scanned documents.

## Features

- 📄 Extract text from regular PDF files
- 🔍 OCR support for scanned PDF documents
- 🌐 Web interface for easy file uploads
- 🔧 RESTful API for programmatic access
- 📱 Swagger UI for API documentation
- 📃 Multi-page PDF support
- 🔒 Secure file handling
- 🐳 Docker support

## Prerequisites

Before running this application, make sure you have the following installed:

- Python 3.x
- Tesseract OCR engine
- poppler-utils (required for pdf2image)

### Installing Tesseract OCR

#### Windows
1. Download the Tesseract installer from the [official GitHub repository](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer
3. Add Tesseract to your system PATH
4. Update the `tesseract_cmd` path in `pdf_reader.py` if necessary

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### macOS
```bash
brew install tesseract
```

### Installing Poppler

#### Windows
Download and install poppler from [poppler releases](http://blog.alivate.com.au/poppler-windows/)

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install poppler-utils
```

#### macOS
```bash
brew install poppler
```

## Installation

### Option 1: Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-text-extractor
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-text-extractor
```

2. Build the Docker image:
```bash
docker build -t pdf-text-extractor .
```

3. Run the container:
```bash
docker run -p 5000:5000 pdf-text-extractor
```

## Configuration

1. Update the Tesseract path in `pdf_reader.py` if necessary:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
```

2. Set your secret key in `app.py`:
```python
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
```

## Usage

### Running the Application

#### Local Installation
1. Start the server:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

#### Docker Installation
1. The application will be available at `http://localhost:5000` after running the container

### Web Interface

1. Open your web browser and navigate to `http://localhost:5000`
2. Click "Choose File" to select a PDF
3. Click "Upload" to process the file
4. View the extracted text on the page

### API Usage

The API provides a RESTful interface for PDF processing. Access the Swagger UI documentation at `http://localhost:5000/` for detailed API information.

#### Example API Request

```python
import requests

url = 'http://localhost:5000/pdf/upload'
files = {'file': open('example.pdf', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## API Endpoints

- `POST /pdf/upload`: Upload and process a PDF file
  - Returns: JSON with page count and extracted text
  - Supports both regular and scanned PDFs

## Project Structure

```
pdf-text-extractor/
├── app.py              # Web interface implementation
├── api.py             # REST API implementation
├── pdf_reader.py      # PDF processing logic
├── requirements.txt   # Project dependencies
├── templates/         # HTML templates
├── uploads/          # Temporary file storage
├── Dockerfile        # Docker configuration
└── .dockerignore     # Docker ignore file
```

## Dependencies

- Flask (3.0.2): Web framework
- PyPDF2 (3.0.1): PDF processing
- Flask-RESTX (1.3.0): API framework
- pytesseract (0.3.10): OCR processing
- pdf2image (1.17.0): PDF to image conversion
- Pillow (10.2.0): Image processing

## Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- Missing files
- PDF processing errors
- OCR processing failures

## Security Features

- Secure filename handling
- Temporary file cleanup
- File extension validation
- Flash messaging for user feedback

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please create an issue in the repository.

 docker run -p 5000:5000 -v "E:/AI/Python/ImageEdit/uploads:/app/uploads" -v "E:/AI/Python/ImageEdit/logs:/app/logs" image-edit