import os
import json
import requests
from typing import List, Dict, Union, Optional
from urllib.parse import urlparse
import logging
import re
from datetime import datetime
from __init__ import root_dir, model_selection
import pdb, traceback
import tiktoken

# Third-party libraries
import docx
import PyPDF2
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text."""
    return len(tokenizer.encode(text))

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate the text to the specified number of tokens."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])

def read_txt_file(filepath: str) -> str:
    """Read a text file and return its content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try reading with a different encoding if UTF-8 fails
        with open(filepath, 'r', encoding='latin-1') as file:
            return file.read()

def read_json_file(filepath: str) -> Dict:
    """Read a JSON file and return its content as a dictionary."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def read_docx_file(filepath: str) -> str:
    """Read a Word document and return its content."""
    doc = docx.Document(filepath)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf_file(filepath: str, max_pages: int = 50) -> str:
    """Read a PDF file and return its content, limiting to max_pages."""
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return '\n'.join([page.extract_text() for page in reader.pages[:max_pages]])

def fetch_web_content(url: str) -> str:
    """Fetch content from a web URL."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Extract text
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    return '\n'.join(chunk for chunk in chunks if chunk)

def infer_title(content: str, filepath: str = '') -> str:
    """Infer a title from the content using an LLM."""
    # Truncate content to first 200 tokens
    truncated_content = truncate_to_token_limit(content, 500)
    
    prompt = f"""
    Based on the following content, generate a concise and informative title. 
    The title should be no more than 20 words long and should capture the main topic or theme of the content.

    Content:
    {truncated_content}

    Title:
    """

    try:
        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with generating document titles."},
            {"role": "user", "content": prompt}
        ]
        
        title = model_selection("gpt-4o", messages=messages, max_tokens=20).strip()
        
        # If LLM fails to generate a title, fall back to the original method
        if not title:
            raise Exception("LLM failed to generate a title")
        
        return title.lower()
    
    except Exception as e:
        logger.warning(f"Error inferring title with LLM: {str(e)}. Falling back to default method.")
        
        # Fall back to the original method
        lines = content.split('\n')
        for line in lines[:5]:
            if len(line.strip()) > 0 and len(line.strip()) < 100:
                return line.strip().lower()
        
        # If no suitable title found, use the filename or URL
        if filepath:
            return os.path.splitext(os.path.basename(filepath))[0].lower()
        return "untitled document"

def extract_metadata(filepath: str) -> Dict[str, str]:
    """Extract metadata from the file."""
    metadata = {}
    
    try:
        # Get file modification time
        mtime = os.path.getmtime(filepath)
        metadata['last_modified'] = datetime.fromtimestamp(mtime).isoformat()
        
        # Get file size
        metadata['file_size'] = f"{os.path.getsize(filepath) / (1024 * 1024):.2f} MB"
        
        # Get file type
        _, ext = os.path.splitext(filepath)
        metadata['file_type'] = ext.lstrip('.')
        
    except Exception as e:
        logger.warning(f"Error extracting metadata from {filepath}: {str(e)}")
    
    return metadata

def prepare_single_document(input_path: str) -> Optional[Dict[str, str]]:
    """Prepare a single document from the given input path."""
    content = ""
    metadata = {}
    
    try:
        if input_path.startswith(('http://', 'https://')):
            content = fetch_web_content(input_path)
            metadata['source'] = input_path
        else:
            _, ext = os.path.splitext(input_path)
            if ext.lower() == '.txt':
                content = read_txt_file(input_path)
            elif ext.lower() == '.json':
                json_content = read_json_file(input_path)
                content = json.dumps(json_content, indent=2)  # Convert JSON to string
            elif ext.lower() == '.docx':
                content = read_docx_file(input_path)
            elif ext.lower() == '.pdf':
                content = read_pdf_file(input_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            metadata = extract_metadata(input_path)
    
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None
    
    title = infer_title(content, input_path)
    
    # Include metadata in content
    metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    full_content = f"{metadata_str}\n\n{content}" if metadata_str else content
    
    logger.info(f"Prepared document: {title} (Content length: {len(full_content)} characters)")
    
    return {
        "title": title,
        "content": full_content
    }

def prepare_documents(input_list: List[str]) -> List[Dict[str, str]]:
    """Prepare documents from a list of file paths and URLs."""
    documents = []
    for input_path in input_list:
        doc = prepare_single_document(input_path)
        if doc:
            documents.append(doc)
    return documents

def main():
    input_folder = os.path.join(root_dir, 'input_files/')
    # Example usage
    input_list = [
        input_folder + "Ray Dalio & Deepak Chopra on Life and Death_interview_20240917_080247.txt",
        input_folder + "document.json",
        input_folder + "bwam071814.docx",
        input_folder + "2021 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf",
        "https://www.linkedin.com/pulse/overview-computer-vision-vivek-murugesan/"
    ]
    
    documents = prepare_documents(input_list)
    
    for doc in documents:
        logger.info(f"Title: {doc['title']}")
        logger.info(f"Content preview: {doc['content'][:100]}...")
        logger.info("---")
    
    return documents

if __name__ == "__main__":
    docs = main()
    # pdb.set_trace()