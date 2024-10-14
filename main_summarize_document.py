import logging
from typing import Dict, Any, List, Tuple, Optional
import json, os
from functools import lru_cache
import tiktoken
from __init__ import model_selection, MAX_TOKENS, root_dir
from classify_document_type import classify_document_type
from generate_basic_summary import generate_basic_summary, prepare_for_retrieval
from retrieve_relevant_context import retrieve_relevant_context
from analyze_significance import analyze_document
from fact_check import fact_check
from integrated_insightful_summarization import generate_insightful_summary_with_refinement
from prepare_document import prepare_documents

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def get_encoding(model: str) -> tiktoken.Encoding:
    """Returns the encoding for the specified model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"No specific tokenizer found for {model}. Using cl100k_base as default.")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in the given text for the specified model."""
    encoding = get_encoding(model)
    return len(encoding.encode(text))

def summarize_document(document: Dict[str, str]) -> Dict[str, Any]:
    """
    Main function to summarize a document.
    
    Args:
    document (Dict[str, str]): A dictionary containing 'title' and 'content' keys.
    
    Returns:
    Dict[str, Any]: A dictionary containing the summarization results.
    """
    title = document.get('title', '')
    content = document.get('content', '')
    
    logger.info(f"Starting summarization for document: {title}")
    logger.info(f"Document length: {count_tokens(content)} tokens")

    try:
        # Step 1: Classify document type
        classification_result = classify_document_type(content)
        if type(classification_result.get('domain')) == list:
            domains = classification_result['domain']
            if "OUTLIER" in domains and len(domains) > 1:
                domains.remove("OUTLIER")
                domain = max(domains, key=domains.count)
            elif "OUTLIER" in domains and len(domains) == 1:
                domains = classification_result['suggested_domains']
                domain = max(domains, key=domains.count)
            else:
                domain = max(domains, key=domains.count) 
        else:
            if classification_result['domain'] == 'OUTLIER':
                domains = classification_result['suggested_domains']
                domain = max(domains, key=domains.count)
            else:
                domain = classification_result['domain']
        logger.info(f"Document classified as: {domain}")

        # Step 2: Generate basic summary
        basic_summaries = generate_basic_summary(content, max_summary_tokens=500)
        logger.info(f"Basic summary generated. Number of chunks: {len(basic_summaries)}")

        retrieval_ready_summaries = prepare_for_retrieval(basic_summaries, title, domain)
        logger.info(f"Prepared {len(retrieval_ready_summaries)} retrieval-ready summaries")

        # Step 3: Retrieve relevant context
        all_context: List[Tuple[str, str, str]] = []
        for item in retrieval_ready_summaries:
            sub_context = retrieve_relevant_context(item['title'], item['content'], domain, title_weight=0.5)
            all_context.extend(sub_context)
        
        # Remove duplicates from context
        all_context = list(set(all_context))
        logger.info(f"Retrieved {len(all_context)} unique context snippets")

        # Step 4: Analyze significance
        analysis = analyze_document(title, content, domain)
        logger.info("Document analyzed for significance")

        # Format context for insightful summary generation
        context = [(item[0], item[1]) for item in all_context]

        # Step 5: Generate insightful summary
        insightful_summary = generate_insightful_summary_with_refinement(
            title=title,
            basic_summary=basic_summaries,
            analysis=analysis,
            domain=domain,
            context=context,
            user_content=content
        )
        logger.info("Insightful summary generated and refined")
        logger.info(f"Insightful summary length: {count_tokens(insightful_summary)} tokens")

        # Step 6: Fact-checking
        fact_checked_summary = fact_check(insightful_summary, context)
        logger.info("Insightful summary fact-checked")
        logger.info(f"Fact-checked summary length: {count_tokens(fact_checked_summary)} tokens")

        return {
            "title": title,
            "domain": domain,
            "basic_summary": basic_summaries,
            "analysis": analysis,
            "insightful_summary": insightful_summary,
            "fact_checked_summary": fact_checked_summary,
        }

    except Exception as e:
        logger.error(f"Error in document summarization: {str(e)}", exc_info=True)
        return {"error": str(e)}

def main(input_list: List[str]):
    # Prepare documents
    docs = prepare_documents(input_list)

    # Summarize each document
    for document in docs:
        result = summarize_document(document)
        
        if "error" in result:
            logger.error(f"Summarization failed: {result['error']}")
        else:
            logger.info("Summarization completed successfully")
            logger.info(json.dumps(result, indent=2))

if __name__ == "__main__":
    input_folder = os.path.join(root_dir, 'input_files/')
    input_list = [
        # input_folder + "Ray Dalio & Deepak Chopra on Life and Death_interview_20240917_080247.txt",
        # input_folder + "document.json",
        input_folder + "bwam071814.docx"
        # input_folder + "2021 The Power of Scale for Parameter-Efficient Prompt Tuning.pdf"，
        # "https://www.linkedin.com/pulse/overview-computer-vision-vivek-murugesan/"
    ]
    main(input_list)