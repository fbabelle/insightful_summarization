import logging
from typing import List, Optional
from __init__ import model_selection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_explanation(verified_summary: str, original_text: str, context: List[str]) -> str:
    """
    Generates an explanation of how the summary was created, including the context used
    and the reasoning behind key points.

    Args:
    verified_summary (str): The fact-checked summary
    original_text (str): The original document text
    context (List[str]): A list of relevant context snippets used in the summarization process

    Returns:
    str: A detailed explanation of the summarization process
    """
    # Prepare the prompt for the explanation generation
    prompt = f"""
    As an AI assistant specializing in explaining complex processes, your task is to generate a detailed explanation 
    of how the following summary was created from the original text, taking into account the provided context.

    Original Text (truncated for brevity):
    {original_text[:1000]}...

    Context Used:
    {' '.join(context[:5])}  # Limit to first 5 context snippets for brevity

    Final Summary:
    {verified_summary}

    Please provide a detailed explanation that covers the following points:
    1. How the original text was analyzed to identify key information.
    2. How the provided context was used to enhance understanding of the text.
    3. The main points that were selected for inclusion in the summary and why.
    4. Any challenges in summarizing this particular text and how they were addressed.
    5. How the summary ensures accuracy and faithfulness to the original text.
    6. The role of fact-checking in refining the summary.

    Your explanation should be clear, informative, and help a user understand the summarization process.

    Explanation:
    """

    try:
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in explaining complex processes clearly and concisely."},
            {"role": "user", "content": prompt}
        ]
        explanation = model_selection("gpt-4o", messages=messages, temperature=0.7)
        return explanation.strip()
    except Exception as e:
        logger.error(f"An error occurred while generating the explanation: {str(e)}")
        return "Unable to generate explanation due to an error."

def format_explanation(explanation: str) -> str:
    """
    Formats the explanation into a more readable structure with sections.

    Args:
    explanation (str): The generated explanation

    Returns:
    str: A formatted explanation with clear sections
    """
    sections = [
        "Analysis of Original Text",
        "Utilization of Context",
        "Selection of Main Points",
        "Challenges and Solutions",
        "Ensuring Accuracy",
        "Role of Fact-Checking"
    ]

    formatted_explanation = "Explanation of the Summarization Process:\n\n"

    for section in sections:
        section_start = explanation.lower().find(section.lower())
        if section_start != -1:
            section_end = explanation.find('\n', section_start)
            if section_end == -1:
                section_end = len(explanation)
            
            formatted_explanation += f"## {section}\n"
            formatted_explanation += explanation[section_start:section_end].strip() + "\n\n"

    return formatted_explanation

def main():
    sample_original_text = "This is a sample original text. It contains important information about a topic."
    sample_context = ["Additional context 1", "Additional context 2"]
    sample_summary = "This is a sample summary of the original text, incorporating the provided context."

    logger.info("Generating explanation...")
    explanation = generate_explanation(sample_summary, sample_original_text, sample_context)
    logger.info("Explanation:")
    logger.info(explanation)

    # logger.info("Formatting explanation...")
    # formatted_explanation = format_explanation(explanation)
    
    # logger.info("Formatted Explanation:")
    # logger.info(formatted_explanation)

if __name__ == "__main__":
    main()