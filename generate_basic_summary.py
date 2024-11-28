import tiktoken
import logging
import json
from typing import Dict, Any, Optional, List
from functools import lru_cache, partial
import multiprocessing
from __init__ import model_selection, MAX_TOKENS, summary_cache, depth_manager, AnalysisDepth

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Returns the encoding for the specified name."""
    return tiktoken.get_encoding(encoding_name)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = get_encoding(encoding_name)
    return len(encoding.encode(string))

def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: float = 0.1) -> List[str]:
    """
    Split the text into overlapping chunks based on token count.
    
    Args:
    text (str): The text to be split into chunks
    max_tokens (int): Maximum number of tokens per chunk
    overlap (float): Fraction of max_tokens to use as overlap between chunks
    
    Returns:
    List[str]: A list of text chunks
    """
    encoding = get_encoding()
    tokens = encoding.encode(text)
    chunks = []
    overlap_tokens = int(max_tokens * overlap)
    
    start = 0
    while start < len(tokens):
        # Determine the end of the current chunk
        end = min(start + max_tokens, len(tokens))
        
        # Extract the chunk
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        
        # Move the start pointer forward, ensuring progress
        start = end - overlap_tokens
        
        # Ensure we don't go backwards
        start = max(start, end - max_tokens)
        
        # Break if we've reached the end
        if end == len(tokens):
            break

    return chunks

def generate_summary_for_chunk(chunk: str, max_summary_tokens: int = 500) -> Dict[str, Any]:
    """Generate a summary and key points for a single chunk of text."""
    try:
        prompt = f"""
        Please provide a concise summary and key points of the following text. 
        The summary should be no more than {max_summary_tokens} tokens long.

        Text to summarize:
        {chunk}

        Respond in JSON format with the following structure:
        {{
            "basic_summary": "Your concise summary here",
            "key_points": ["Key point 1", "Key point 2", "Key point 3", ...]
        }}
        """

        messages = [
            {"role": "system", "content": "You are an AI assistant skilled in creating concise and accurate summaries with key points."},
            {"role": "user", "content": prompt}
        ]
        
        response = model_selection("gpt-4o", messages=messages, output_json=True, temperature=0.5)
        result = json.loads(response.strip())
        
        logger.info(f"Summary generated successfully. Length: {num_tokens_from_string(result['basic_summary'])} tokens")
        return result

    except Exception as e:
        logger.error(f"An error occurred during summary generation: {str(e)}")
        return {"basic_summary": f"Error: Unable to generate summary. {str(e)}", "key_points": []}

# def generate_basic_summary(document: str, max_summary_tokens: int = 2000) -> List[Dict[str, Any]]:
def generate_basic_summary(document: str, doc_type: str, time_constraint: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate basic summaries and key points for the given document, handling long texts with chunking and parallelization.
    
    Args:
    document (str): The text of the document to be summarized
    
    Returns:
    List[Dict[str, Any]]: A list of dictionaries, each containing a basic summary and key points for a chunk of the document
    """
    
    # Determine appropriate depth
    depth = depth_manager.determine_depth(doc_type, len(document), time_constraint)
    config = depth_manager.get_config(depth)

    # Cache key based on the document hash and max summary tokens
    cache_key = f"summary:{hash(document)}:{config.max_summary_tokens}"
    
    # Check cache first
    cached_result = summary_cache.get(cache_key)
    if cached_result is not None:
        logger.info("Cache hit for basic summary generation")
        return cached_result

    # chunks = chunk_text(document, max_tokens=MAX_TOKENS, overlap=0.1)

    # Use configuration for chunking
    chunks = chunk_text(
        document, 
        max_tokens=config.chunk_size, 
        overlap=config.overlap
    )
    
    # # Create a pool of workers
    # with multiprocessing.Pool() as pool:
    #     # Map the chunks to the generate_summary_for_chunk function in parallel
    #     results = pool.map(generate_summary_for_chunk, chunks, max_summary_tokens)
    
    # Process based on configuration
    if config.parallel_processing:
        with multiprocessing.Pool() as pool:
            results = pool.map(
                partial(generate_summary_for_chunk, 
                       max_summary_tokens=config.max_summary_tokens),
                chunks
            )
    else:
        results = [
            generate_summary_for_chunk(chunk, max_summary_tokens=config.max_summary_tokens) 
            for chunk in chunks
        ]

    # Cache the result
    token_count = sum(num_tokens_from_string(str(result)) for result in results)
    summary_cache.put(cache_key, results, token_count)

    return results

def prepare_for_retrieval(summaries: List[Dict[str, Any]], title: str, classification: str) -> List[Dict[str, Any]]:
    """
    Prepare the summaries for the retrieval step by combining summaries, key points, title, and classification.
    
    Args:
    summaries (List[Dict[str, Any]]): List of dictionaries containing basic summaries and key points
    title (str): The title of the document
    classification (str): The classification result of the document
    
    Returns:
    List[Dict[str, Any]]: A list of dictionaries formatted for the retrieval step
    """
    retrieval_ready_summaries = []
    
    for i, summary in enumerate(summaries, 1):
        formatted_summary = {
            "title": title,
            "classification": classification,
            "content": f"Summary (Part {i}/{len(summaries)}):\n{summary['basic_summary']}\n\nKey Points:\n" + 
                       "\n".join(f"- {point}" for point in summary['key_points'])
        }
        retrieval_ready_summaries.append(formatted_summary)
    
    return retrieval_ready_summaries

def main():
    # Example document text
    document = """
    Climate change is one of the most pressing issues facing our planet today. It refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities, especially the burning of fossil fuels.

    These activities release greenhouse gases into the atmosphere, primarily carbon dioxide and methane. These gases trap heat from the sun, causing the Earth's average temperature to rise. This phenomenon is known as the greenhouse effect.

    The impacts of climate change are far-reaching and significant. They include more frequent and severe weather events, such as hurricanes, droughts, and heatwaves. Rising sea levels threaten coastal communities and islands. Changes in temperature and precipitation patterns affect agriculture and food security.

    Moreover, climate change poses risks to human health, biodiversity, and economic stability. It exacerbates existing social and economic inequalities, as vulnerable populations often bear the brunt of its effects.

    Addressing climate change requires a multi-faceted approach. This includes transitioning to renewable energy sources, improving energy efficiency, protecting and restoring ecosystems, and adapting to the changes already set in motion. International cooperation, as seen in agreements like the Paris Accord, is crucial in coordinating global efforts to mitigate climate change.

    While the challenge is immense, there is still time to act. Every individual, business, and government has a role to play in reducing greenhouse gas emissions and building a sustainable future. The choices we make today will determine the world we leave for future generations.
    """
    
    # In a real scenario, these would be passed from the main workflow script
    document_title = "The Impact and Challenges of Climate Change"
    document_classification = "scientific_research_paper"
    
    summaries = generate_basic_summary(document, document_classification)
    logger.info("Summaries and Key Points:")
    for i, summary in enumerate(summaries, 1):
        logger.info(f"Chunk {i}:")
        logger.info(json.dumps(summary, indent=2))
    
    retrieval_ready_summaries = prepare_for_retrieval(summaries, document_title, document_classification)
    logger.info("\nRetrieval-Ready Summaries:")
    for i, summary in enumerate(retrieval_ready_summaries, 1):
        logger.info(f"Retrieval-Ready Chunk {i}:")
        logger.info(json.dumps(summary, indent=2))

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Necessary for Windows compatibility
    main()