import json
from typing import List, Dict, Union
from collections import Counter
import tiktoken
import multiprocessing
from functools import partial
from __init__ import domains, model_selection, logger, MAX_TOKENS, classification_cache

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(tokenizer.encode(string))

def chunk_text(text: str, max_tokens: int = 1000, overlap: float = 0.1) -> List[str]:
    """
    Split the text into overlapping chunks based on token count.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        
        if start > 0:
            start = max(start - int(max_tokens*overlap), 0)
        
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        
        start = end

        if end == len(tokens):
            break

    return chunks

def classify_chunk(chunk: str, title: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Classify a single chunk of text using the selected model.
    """
    prompt = f"""
    Classify the following text chunk into one of these categories: {', '.join(domains)}.
    
    Title: {title}
    Text Chunk: {chunk[:500]}...  # Truncated for brevity

    Respond in JSON format with the following structure:
    {{
        "classification": "domain_name or OUTLIER",
        "suggested_domain": "a single suggested domain name if OUTLIER, and it should be one of the available domains based on relavance, otherwise null"
    }}
    """

    try:
        response = model_selection(
            model_name='gpt-4o',
            messages=[
                {"role": "system", "content": "You are a document classification assistant. Respond only in the specified JSON format."},
                {"role": "user", "content": prompt}
            ],
            output_json=True,
            temperature=0.1
        )
        
        classification_result = json.loads(response)
        logger.info(f"Chunk classification result: {classification_result}")
        return classification_result

    except Exception as e:
        logger.error(f"An error occurred while classifying the chunk: {str(e)}")
        return {"classification": "ERROR", "suggested_domain": None, "error": str(e)}

def classify_document_type(document: str, title: str = "") -> Dict[str, Union[str, List[str], float]]:
    """
    Classify the type of the input document using the selected model with chunking and voting.
    """

    # Generate a unique cache key for the document and title
    cache_key = f"classification:{hash(document + title)}"
    
    # Check cache first
    cached_result = classification_cache.get(cache_key)
    if cached_result is not None:
        logger.info("Cache hit for document classification")
        return cached_result

    chunks = chunk_text(document, max_tokens=MAX_TOKENS // 2)
    # chunks = chunk_text(document, max_tokens=30)
    
    # Create a pool of workers
    with multiprocessing.Pool() as pool:
        # Use partial to create a function with fixed title parameter
        classify_func = partial(classify_chunk, title=title)
        # Map the chunks to the classify_func in parallel
        chunk_classifications = pool.map(classify_func, chunks)
    
    # Count the votes for each domain
    votes = Counter([c['classification'] for c in chunk_classifications if 'error' not in c])
    total_votes = sum(votes.values())

    # Sort domains by vote count
    sorted_domains = sorted(votes.items(), key=lambda x: x[1], reverse=True)

    # Prepare the result
    result = {}

    # Check if the top domain has more than 50% of votes
    if sorted_domains and sorted_domains[0][1] > total_votes * 0.5:
        top_domain = sorted_domains[0][0]
        result = {
            "domain": top_domain,
            "confidence": "High" if sorted_domains[0][1] > total_votes * 0.7 else "Medium",
            "vote_percentage": sorted_domains[0][1] / total_votes
        }
    else:
        # Find domains that together exceed 70% of votes
        cumulative_votes = 0
        selected_domains = []
        for domain, vote_count in sorted_domains:
            selected_domains.append(domain)
            cumulative_votes += vote_count
            if cumulative_votes > total_votes * 0.7:
                break

        result = {
            "domain": selected_domains,
            "confidence": "Low",
            "vote_percentages": {domain: votes[domain] / total_votes for domain in selected_domains}
        }

    # Handle outliers
    if "OUTLIER" in result.get("domain", []) or result.get("domain") == "OUTLIER":
        suggested_domains = [c['suggested_domain'] for c in chunk_classifications if c['classification'] == "OUTLIER" and c['suggested_domain']]
        result["suggested_domains"] = list(set(suggested_domains))

    # Cache the result
    token_count = num_tokens_from_string(str(result))
    classification_cache.put(cache_key, result, token_count)

    logger.info(f"Final classification result: {result}")
    return result

def main():
    logger.info("Starting document classification")
    # Test the classifier with a sample document
    sample_title = "Quantum Computing: A New Era in Technology"
    sample_document = """
    Quantum computing represents a paradigm shift in computational capabilities, promising to revolutionize fields ranging from cryptography to drug discovery. Unlike classical computers that use bits, quantum computers leverage quantum bits or qubits, which can exist in multiple states simultaneously due to the principle of superposition.

    Recent advancements have shown remarkable progress in quantum supremacy, where quantum computers perform tasks intractable for classical systems. Google's 53-qubit Sycamore processor claimed this milestone in 2019, completing a specific calculation in 200 seconds that would take the world's most powerful supercomputer 10,000 years.

    However, challenges remain in scaling quantum systems and maintaining qubit coherence. Researchers are exploring various qubit implementations, including superconducting circuits, trapped ions, and topological qubits, each with unique advantages and obstacles.

    The potential applications of quantum computing are vast. In the pharmaceutical industry, quantum simulations could accelerate drug discovery by modeling complex molecular interactions. In finance, quantum algorithms could optimize portfolio management and risk assessment. Quantum-resistant cryptography is becoming crucial as quantum computers threaten current encryption methods.

    As quantum computing evolves, it's essential to consider its ethical implications. The technology could exacerbate global inequalities if access is limited to wealthy nations and corporations. Moreover, the potential to break current encryption systems raises significant security concerns.

    In conclusion, while quantum computing is still in its early stages, its rapid development suggests we are on the cusp of a new technological era. The coming decades will likely see quantum computers move from research labs to practical applications, fundamentally changing our approach to complex problem-solving across various domains.
    """
    
    classification_result = classify_document_type(sample_document, sample_title)
    logger.info(f"Classification Result: {json.dumps(classification_result, indent=2)}")

    # Example of how to use the classification result in the retrieval script
    if isinstance(classification_result["domain"], str):
        domain = classification_result["domain"]
        logger.info(f"Passing to retrieval script - Domain: {domain}, Title: {sample_title}")
        # retrieve_relevant_context(sample_document, domain, title=sample_title)  # Uncomment and use your actual retrieval function
    else:
        logger.info(f"Multiple potential domains detected: {classification_result['domain']}")
        for domain in classification_result["domain"]:
            logger.info(f"Passing to retrieval script - Domain: {domain}, Title: {sample_title}")
            # retrieve_relevant_context(sample_document, domain, title=sample_title)  # Uncomment and use your actual retrieval function

    if "suggested_domains" in classification_result:
        logger.info(f"Outlier detected. Suggested domains: {classification_result['suggested_domains']}")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Necessary for Windows compatibility
    main()