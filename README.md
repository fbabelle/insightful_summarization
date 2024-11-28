# Insightful Document Summarization

An AI-powered document summarization system that generates comprehensive, context-aware summaries through multi-stage processing and intelligent analysis.

## Overview

This system processes documents through multiple stages:
1. Document classification
2. Basic summarization
3. Context retrieval
4. Significance analysis
5. Insightful summary generation
6. Fact checking

## Usage

from insightful_summarization import summarize_document

# Prepare your document
document = {
    "title": "Your Document Title",
    "content": "Your document content..."
}

# Generate summary
result = summarize_document(document)

# Access different summary components
print(result["domain"])              # Document classification
print(result["basic_summary"])       # Initial summary
print(result["analysis"])            # Document analysis
print(result["insightful_summary"])  # Enhanced summary
print(result["fact_checked_summary"]) # Verified summary

## Features

- **Document Classification**: Automatically identifies document type and domain
- **Multi-Stage Processing**: Generates summaries through multiple refinement stages
- **Context Integration**: Incorporates relevant background information
- **Fact Checking**: Verifies generated summaries against source material
- **Token Management**: Optimized for various model token limits
- **Logging**: Comprehensive logging for process tracking

## Requirements

- Python 3.7+
- tiktoken

## Project Structure

insightful_summarization/
├── main_summarize_document.py
├── classify_document_type.py
├── generate_basic_summary.py
├── retrieve_relevant_context.py
├── analyze_significance.py
├── fact_check.py
├── integrated_insightful_summarization.py
└── prepare_document.py
