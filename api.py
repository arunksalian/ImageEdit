from flask_restx import Api, Resource, fields
from flask import request
import os
from werkzeug.utils import secure_filename
from pdf_reader import read_pdf

# Initialize API
api = Api(
    title='PDF Reader API',
    version='1.0',
    description='A simple API for reading PDF files'
)

# Define namespace
ns = api.namespace('pdf', description='PDF operations')

# Define models for Swagger documentation
upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type='FileStorage', required=True, help='PDF file to upload')

pdf_content_model = api.model('PDFContent', {
    'total_pages': fields.Integer(description='Total number of pages in the PDF'),
    'pages': fields.List(fields.String(description='Content of each page'))
})

@ns.route('/upload')
class PDFUpload(Resource):
    @ns.expect(upload_parser)
    @ns.response(200, 'Success', pdf_content_model)
    @ns.response(400, 'Invalid file type')
    @ns.response(500, 'Server error')
    def post(self):
        """Upload and read a PDF file"""
        if 'file' not in request.files:
            return {'message': 'No file provided'}, 400
        
        file = request.files['file']
        
        if file.filename == '':
            return {'message': 'No file selected'}, 400
        
        if not file.filename.lower().endswith('.pdf'):
            return {'message': 'Only PDF files are allowed'}, 400
        
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            
            # Read PDF content
            content = read_pdf(filepath)
            
            # Format response
            total_pages = int(content[0].split(': ')[1])
            pages = content[1:]  # Skip the total pages line
            
            return {
                'total_pages': total_pages,
                'pages': pages
            }
            
        except Exception as e:
            return {'message': str(e)}, 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath) 