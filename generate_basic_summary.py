import tiktoken
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

from __init__ import model_selection

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

def truncate_document(document: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Truncates the document to the specified maximum number of tokens."""
    encoding = get_encoding(encoding_name)
    truncated_tokens = encoding.encode(document)[:max_tokens]
    return encoding.decode(truncated_tokens)

def generate_basic_summary(document: str, max_tokens: int = 150, max_input_tokens: int = 4000) -> str:
    """
    Generate a basic summary of the given document using GPT-4.
    
    Args:
    document (str): The text of the document to be summarized
    max_tokens (int): Maximum number of tokens for the summary (default: 150)
    max_input_tokens (int): Maximum number of input tokens (default: 4000)
    
    Returns:
    str: A basic summary of the document
    """
    try:
        # Truncate the document if it's too long
        if num_tokens_from_string(document) > max_input_tokens:
            document = truncate_document(document, max_input_tokens)
            logger.info(f"Document truncated to {max_input_tokens} tokens")

        prompt = f"""
        Please provide a concise summary of the following text. The summary should be no more than {max_tokens} tokens long and should capture the main points of the text.

        Text to summarize:
        {document}

        Summary:
        """

        messages = [
            {"role": "system", "content": "You are an AI assistant skilled in creating concise and accurate summaries."},
            {"role": "user", "content": prompt}
        ]
        
        response = model_selection("gpt-4o", messages=messages, output_json=False, temperature=0.5)
        summary = response.strip()
        
        logger.info(f"Summary generated successfully. Length: {len(summary)} characters")
        return summary

    except Exception as e:
        logger.error(f"An error occurred during summary generation: {str(e)}")
        return f"Error: Unable to generate summary. {str(e)}"

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
    
    summary = generate_basic_summary(document)
    logger.info("Basic Summary:")
    logger.info(summary)

if __name__ == "__main__":
    main()