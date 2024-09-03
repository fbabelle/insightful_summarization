import textwrap
import logging
from typing import Dict, List, Optional
from functools import lru_cache
import tiktoken
from __init__ import model_selection

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

def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in the given text for the specified model."""
    encoding = get_encoding(model)
    return len(encoding.encode(text))

def combine(text: str, context: List[str], max_tokens: int = 3000) -> str:
    """
    Combines the original text with retrieved context.
    
    Args:
    text (str): The original document text
    context (List[str]): A list of relevant context snippets
    max_tokens (int): Maximum number of tokens in the combined text
    
    Returns:
    str: The augmented input text
    """
    combined = text
    for snippet in context:
        if count_tokens(combined + snippet, "gpt-4") > max_tokens:
            break
        combined += f"\n\nRelevant Context: {snippet}"
    return combined

def construct_prompt(basic_summary: str, analysis: Dict, domain: str) -> str:
    """
    Constructs a prompt for the LLM to generate an insightful summary.
    
    Args:
    basic_summary (str): A basic summary of the original text
    analysis (Dict): The output from the domain-specific analysis function
    domain (str): The type of document ('scientific_paper', 'news', or 'article')
    
    Returns:
    str: A prompt for the LLM
    """
    prompts = {
        'scientific_paper': textwrap.dedent("""
            You are an expert scientific communicator. Your task is to create an insightful summary of a scientific paper.
            Use the following information to craft your summary:

            Basic Summary: {basic_summary}

            Key Contributions:
            {contributions}

            Relation to Previous Work:
            {previous_work}

            Potential Impact:
            {impact}

            Generate a comprehensive yet concise summary that:
            1. Clearly states the main findings and their significance
            2. Places the research in the context of the field
            3. Explains the potential impact and applications of the work
            4. Identifies any limitations or areas for future research

            Your summary should be informative to both experts and informed lay readers.
        """),
        'news': textwrap.dedent("""
            You are an experienced journalist and analyst. Your task is to create an insightful summary of a news article.
            Use the following information to craft your summary:

            Basic Summary: {basic_summary}

            Key Events:
            {key_events}

            Broader Context:
            {context}

            Potential Implications:
            {implications}

            Generate a comprehensive yet concise summary that:
            1. Clearly outlines the key events and their immediate significance
            2. Places the news in a broader context (historical, social, political, etc.)
            3. Explains why this is newsworthy and its potential impact
            4. Presents any relevant controversies or differing viewpoints

            Your summary should be informative and provide deeper insights than a typical news report.
        """),
        'article': textwrap.dedent("""
            You are a skilled literary analyst and critic. Your task is to create an insightful summary of an article or opinion piece.
            Use the following information to craft your summary:

            Basic Summary: {basic_summary}

            Main Argument:
            {main_argument}

            Author's Stance:
            {stance}

            Rhetorical Strategies:
            {rhetorical_strategies}

            Generate a comprehensive yet concise summary that:
            1. Clearly states the main argument or thesis of the article
            2. Explains the author's stance and perspective
            3. Analyzes the rhetorical strategies and persuasive techniques used
            4. Evaluates the effectiveness of the argument and its potential impact

            Your summary should provide a deeper understanding of both the content and the craft of the article.
        """)
    }
    
    prompt_template = prompts.get(domain, prompts['article'])
    return prompt_template.format(basic_summary=basic_summary, **analysis)

def generate_insightful_summary(prompt: str, max_tokens: int = 1000, model: str = "gpt-4") -> Optional[str]:
    """
    Generates an insightful summary using the specified LLM.
    
    Args:
    prompt (str): The constructed prompt to guide the summary generation
    max_tokens (int): Maximum number of tokens in the generated summary
    model (str): The LLM model to use (default is "gpt-4")
    
    Returns:
    Optional[str]: The generated insightful summary, or None if an error occurs
    """
    try:
        prompt_tokens = count_tokens(prompt, model)
        available_tokens = 8192 - prompt_tokens  # Assuming GPT-4's context length
        max_tokens = min(max_tokens, available_tokens)

        messages = [
            {"role": "system", "content": "You are an expert summarizer capable of providing insightful, well-structured summaries."},
            {"role": "user", "content": prompt}
        ]

        summary = model_selection(model, messages=messages, temperature=0.7)
        logger.info(f"Summary generated successfully. Length: {len(summary)} characters")
        return summary.strip()

    except Exception as e:
        logger.error(f"An error occurred while generating the summary: {str(e)}")
        return None

def refine_summary(summary: str, original_text: str, model: str = "gpt-4") -> str:
    """
    Refines the generated summary to ensure accuracy and coherence.
    
    Args:
    summary (str): The initially generated summary
    original_text (str): The original document text
    model (str): The LLM model to use (default is "gpt-4")
    
    Returns:
    str: The refined summary
    """
    refinement_prompt = f"""
    You are an expert editor. Your task is to refine the following summary to ensure it accurately represents the original text, 
    is coherent, and provides insightful analysis. Here are the original text and the generated summary:

    Original Text:
    {original_text}

    Generated Summary:
    {summary}

    Please refine the summary, focusing on:
    1. Accuracy: Ensure all facts and claims align with the original text.
    2. Coherence: Improve the flow and structure of the summary.
    3. Insight: Enhance any analytical or contextual elements.
    4. Conciseness: Trim any unnecessary information while retaining key insights.

    Provide the refined summary:
    """

    try:
        messages = [
            {"role": "system", "content": "You are an expert editor skilled in refining summaries."},
            {"role": "user", "content": refinement_prompt}
        ]

        refined_summary = model_selection(model, messages=messages, temperature=0.3)
        logger.info(f"Summary refined successfully. Length: {len(refined_summary)} characters")
        return refined_summary.strip()

    except Exception as e:
        logger.error(f"An error occurred while refining the summary: {str(e)}")
        return summary  # Return the original summary if refinement fails

def generate_insightful_summary_with_refinement(
    original_text: str,
    basic_summary: str,
    analysis: Dict,
    domain: str,
    context: List[str],
    max_tokens: int = 1000,
    model: str = "gpt-4"
) -> Optional[str]:
    """
    Generates an insightful summary and refines it for improved quality.
    
    Args:
    original_text (str): The original document text
    basic_summary (str): A basic summary of the original text
    analysis (Dict): The output from the domain-specific analysis function
    domain (str): The type of document ('scientific_paper', 'news', or 'article')
    context (List[str]): A list of relevant context snippets
    max_tokens (int): Maximum number of tokens in the generated summary
    model (str): The LLM model to use (default is "gpt-4")
    
    Returns:
    Optional[str]: The refined insightful summary, or None if an error occurs
    """
    logger.info("Starting insightful summary generation process")
    
    combined_text = combine(original_text, context)
    prompt = construct_prompt(basic_summary, analysis, domain)
    
    initial_summary = generate_insightful_summary(prompt, max_tokens, model)
    if initial_summary:
        refined_summary = refine_summary(initial_summary, combined_text, model)
        logger.info("Insightful summary generation and refinement completed successfully")
        return refined_summary
    else:
        logger.error("Failed to generate initial summary")
        return None

def main():
    # Example usage
    original_text = "This is a sample scientific paper about climate change impacts."
    basic_summary = "The paper discusses climate change effects on biodiversity."
    analysis = {
        "contributions": "1. New climate model. 2. Biodiversity impact assessment.",
        "previous_work": "Builds on IPCC reports and recent ecological studies.",
        "impact": "Potential for improved conservation strategies."
    }
    domain = "scientific_paper"
    context = [
        "Recent studies show accelerated species loss due to climate change.",
        "Global temperatures have risen by 1.1°C since the pre-industrial era."
    ]

    final_summary = generate_insightful_summary_with_refinement(
        original_text, basic_summary, analysis, domain, context
    )

    if final_summary:
        logger.info("Final Insightful Summary:")
        logger.info(final_summary)
    else:
        logger.error("Failed to generate insightful summary")

if __name__ == "__main__":
    main()