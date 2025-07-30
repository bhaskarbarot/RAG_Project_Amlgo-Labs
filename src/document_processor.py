"""Document processing utilities"""
import PyPDF2
import re
import json
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize

class DocumentProcessor:
    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        text = re.sub(r'([.,!?;]){2,}', r'\1', text)
        return text.strip()
