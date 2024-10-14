import textwrap
import logging
import json
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import tiktoken
from __init__ import model_selection, MAX_TOKENS

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

def truncate_to_token_limit(text: str, max_tokens: int, model: str) -> str:
    """Truncate the text to fit within the specified token limit."""
    encoding = get_encoding(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

def process_context(context: List[Tuple[str, str]], user_title: str, user_content: str, model: str = "gpt-4o") -> str:
    """
    Process and summarize the context to extract core information related to the user's document.
    """
    context_prompt = f"""
    Analyze the following retrieved documents in relation to the user's document. 
    Extract and summarize the most relevant and core information that provides historical 
    and background context to the user's document. Focus on information that will help 
    create an insightful summary of the user's document.

    User's Document Title: {user_title}
    User's Document Content (excerpt): {truncate_to_token_limit(user_content, 2000, model)}

    Retrieved Documents:
    """
    
    for title, content in context:
        context_prompt += f"\nTitle: {title}\nContent (excerpt): {truncate_to_token_limit(content, 2000, model)}\n"

    context_prompt += """
    Provide a comprehensive summary of the core information from these documents that relates to and provides context for the user's document. 
    Focus on:
    1. Historical background
    2. Related research or events
    3. Key concepts or theories mentioned in multiple documents
    4. Contrasting viewpoints or debates in the field
    5. Recent developments or trends relevant to the user's document

    Your summary should be explicit and verbose, providing rich context for the user's document.
    """

    try:
        messages = [
            {"role": "system", "content": "You are an expert at extracting and summarizing core information from multiple sources."},
            {"role": "user", "content": context_prompt}
        ]

        core_info = model_selection(model, messages=messages, max_tokens=4000, temperature=0.3)
        token_count = count_tokens(core_info, model)
        logger.info(f"Core information extracted successfully. Length: {token_count} tokens")
        return core_info.strip()

    except Exception as e:
        logger.error(f"An error occurred while processing context: {str(e)}")
        return ""

def align_keys(analysis_keys: List[str], format_keys: List[str], model: str = "gpt-4o") -> Dict[str, str]:
    """
    Use LLM to align keys from the analysis dictionary with keys from the format_dict.
    
    Args:
    analysis_keys (List[str]): List of keys from the analysis dictionary
    format_keys (List[str]): List of keys from the format_dict
    model (str): The LLM model to use
    
    Returns:
    Dict[str, str]: A dictionary mapping analysis keys to format keys
    """
    prompt = f"""
    You are an expert in natural language understanding and semantic similarity. Your task is to align two sets of keys based on their semantic meaning and likely content. Some keys may not have a match, and that's okay.

    Set 1 (Analysis Keys): {', '.join(analysis_keys)}
    Set 2 (Format Keys): {', '.join(format_keys)}

    Please provide a JSON object where the keys are from Set 1, and the values are the most semantically similar keys from Set 2. If there's no good match, use null as the value.

    Example output format:
    {{
        "key_from_set1": "matching_key_from_set2",
        "another_key_from_set1": null
    }}

    Ensure your response is a valid JSON object.
    """

    try:
        messages = [
            {"role": "system", "content": "You are an AI assistant skilled in understanding semantic similarities between words and phrases."},
            {"role": "user", "content": prompt}
        ]

        response = model_selection(model, messages=messages, temperature=0.3, output_json=True)
        alignment = json.loads(response)
        return alignment

    except json.JSONDecodeError:
        logger.error("Failed to parse JSON response from the model")
        return {}
    except Exception as e:
        logger.error(f"An error occurred while aligning keys: {str(e)}")
        return {}

def construct_prompt(basic_summary: str, analysis: Dict, domain: str, core_context: str, model: str = "gpt-4o") -> str:
    """
    Constructs a prompt for the LLM to generate an insightful summary.
    """
    prompts = {
        'scientific_research_paper': textwrap.dedent("""
            You are an expert scientific communicator. Create an insightful summary of a scientific paper using the following information:

            Basic Summary: {basic_summary}

            Key Contributions: {contributions}
            Relation to Previous Work: {previous_work}
            Potential Impact: {impact}

            Core Context and Background Information:
            {core_context}

            Generate a comprehensive yet concise summary (maximum 500 tokens) that:
            1. Clearly states the main findings and their significance
            2. Places the research in the context of the field, using the provided background information
            3. Explains the potential impact and applications of the work
            4. Identifies any limitations or areas for future research

            Your summary should be informative to both experts and informed lay readers.
        """),
        'news': textwrap.dedent("""
            You are an experienced journalist and analyst. Create an insightful summary of a news article using the following information:

            Basic Summary: {basic_summary}

            Key Events: {key_events}
            Broader Context: {context}
            Potential Implications: {implications}

            Core Context and Background Information:
            {core_context}

            Generate a comprehensive yet concise summary (maximum 500 tokens) that:
            1. Clearly outlines the key events and their immediate significance
            2. Places the news in a broader context (historical, social, political, etc.) using the provided background information
            3. Explains why this is newsworthy and its potential impact
            4. Presents any relevant controversies or differing viewpoints

            Your summary should be informative and provide deeper insights than a typical news report.
        """),
        'article': textwrap.dedent("""
            You are a skilled literary analyst and critic. Create an insightful summary of an article or opinion piece using the following information:

            Basic Summary: {basic_summary}

            Main Argument: {main_argument}
            Author's Stance: {stance}
            Rhetorical Strategies: {rhetorical_strategies}

            Core Context and Background Information:
            {core_context}

            Generate a comprehensive yet concise summary (maximum 500 tokens) that:
            1. Clearly states the main argument or thesis of the article
            2. Explains the author's stance and perspective
            3. Analyzes the rhetorical strategies and persuasive techniques used
            4. Evaluates the effectiveness of the argument and its potential impact
            5. Places the article in the broader context of the topic or debate, using the provided background information

            Your summary should provide a deeper understanding of both the content and the craft of the article.
        """)
    }
    
    prompt_template = prompts.get(domain, prompts['article'])
    
    # Create a dictionary with default values for all possible placeholders
    format_dict = {
        'basic_summary': basic_summary,
        'core_context': core_context,
        'contributions': 'Not specified',
        'previous_work': 'Not specified',
        'impact': 'Not specified',
        'key_events': 'Not specified',
        'context': 'Not specified',
        'implications': 'Not specified',
        'main_argument': 'Not specified',
        'stance': 'Not specified',
        'rhetorical_strategies': 'Not specified'
    }
    
    # Align keys between analysis and format_dict
    alignment = align_keys(list(analysis.keys()), list(format_dict.keys()), model)
    
    # Update format_dict with aligned values from analysis
    for analysis_key, format_key in alignment.items():
        if format_key and analysis_key in analysis:
            format_dict[format_key] = analysis[analysis_key]
    
    # Use the updated format_dict to fill in the prompt template
    return prompt_template.format(**format_dict)

def generate_insightful_summary(prompt: str, model: str = "gpt-4o") -> Optional[str]:
    """
    Generates an insightful summary using the specified LLM.
    """
    try:
        messages = [
            {"role": "system", "content": "You are an expert summarizer capable of providing insightful, well-structured summaries."},
            {"role": "user", "content": prompt}
        ]

        summary = model_selection(model, messages=messages, max_tokens=1000, temperature=0.7)
        token_count = count_tokens(summary, model)
        logger.info(f"Summary generated successfully. Length: {token_count} tokens")
        return summary.strip()

    except Exception as e:
        logger.error(f"An error occurred while generating the summary: {str(e)}")
        return None

def generate_insightful_summary_with_refinement(
    title: str,
    basic_summary: str,
    analysis: Dict,
    domain: str,
    context: List[Tuple[str, str]],
    user_content: str,
    model: str = "gpt-4o"
) -> Optional[str]:
    """
    Generates an insightful summary based on the provided information and context.
    """
    logger.info("Starting insightful summary generation process")
    
    core_context = process_context(context, title, user_content, model)
    prompt = construct_prompt(basic_summary, analysis, domain, core_context)
    
    final_summary = generate_insightful_summary(prompt, model)
    if final_summary:
        logger.info("Insightful summary generation completed successfully")
        return final_summary
    else:
        logger.error("Failed to generate summary")
        return None

def main():
    # Example usage
    title = "Climate Change Impacts on Global Biodiversity"
    basic_summary = "The paper discusses the effects of climate change on global biodiversity patterns."
    analysis = {
        "key_findings": "1. New climate-biodiversity model. 2. Global biodiversity impact assessment.",
        "literature_review": "Builds on IPCC reports and recent ecological studies.",
        "significance": "Potential for improved conservation strategies and policy making."
    }
    domain = "scientific_research_paper"
    context = [
        ("Recent Trends in Global Biodiversity Loss", "This paper examines the accelerating rate of species extinction..."),
        ("Climate Change: A Comprehensive Review", "An overview of climate change causes, effects, and mitigation strategies..."),
        ("Conservation Strategies in the Anthropocene", "Discussion of novel approaches to biodiversity conservation in the face of rapid global change...")
    ]
    user_content = "This study presents a comprehensive analysis of climate change impacts on global biodiversity..."

    final_summary = generate_insightful_summary_with_refinement(
        title, basic_summary, analysis, domain, context, user_content
    )

    if final_summary:
        logger.info("Final Insightful Summary:")
        logger.info(final_summary)
        logger.info(f"Final summary token count: {count_tokens(final_summary, 'gpt-4o')}")
    else:
        logger.error("Failed to generate insightful summary")

if __name__ == "__main__":
    main()