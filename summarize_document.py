import logging
from typing import Dict, Any, Optional
import tiktoken
from __init__ import model_selection
from classify_document_type import classify_document_type
from retrieve_relevant_context import retrieve_relevant_context
from analyze_significance import analyze_scientific_contributions, analyze_news_significance, analyze_author_intent
from fact_check import fact_check, SimpleTrustedKnowledgeBase
from collect_human_feedback import collect_human_feedback
from integrated_insightful_summarization import generate_insightful_summary_with_refinement

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def generate_basic_summary(document: str, max_tokens: int = 150) -> str:
    """Generate a basic summary of the given document."""
    max_input_tokens = 4000
    if num_tokens_from_string(document) > max_input_tokens:
        encoding = tiktoken.get_encoding("cl100k_base")
        document = encoding.decode(encoding.encode(document)[:max_input_tokens])

    prompt = f"""
    Provide a concise summary of the following text in no more than {max_tokens} tokens:

    {document}

    Summary:
    """

    try:
        messages = [
            {"role": "system", "content": "You are an AI assistant skilled in creating concise and accurate summaries."},
            {"role": "user", "content": prompt}
        ]
        response = model_selection("gpt-4o", messages=messages, temperature=0.7)
        return response.strip()
    except Exception as e:
        logger.error(f"Error in basic summary generation: {str(e)}")
        return f"Error: Unable to generate summary. {str(e)}"

def summarize_document(document: str) -> Dict[str, Any]:
    """Main function to summarize a document."""
    try:
        # Step 1: Classify document type
        domain = classify_document_type(document)
        logger.info(f"Document classified as: {domain}")

        # Step 2: Generate basic summary
        basic_summary = generate_basic_summary(document)
        logger.info("Basic summary generated")

        # Step 3: Retrieve relevant context
        context = retrieve_relevant_context(document, domain)
        logger.info(f"Retrieved {len(context)} relevant context snippets")

        # Step 4: Analyze significance
        if domain == 'scientific_paper':
            analysis = analyze_scientific_contributions(document)
        elif domain == 'news':
            analysis = analyze_news_significance(document)
        else:
            analysis = analyze_author_intent(document)
        logger.info("Document significance analyzed")

        # Ensure analysis is a dictionary and has required keys
        if not isinstance(analysis, dict):
            analysis = {"main_points": str(analysis)}
        
        required_keys = {
            'scientific_paper': ['contributions', 'previous_work', 'impact'],
            'news': ['key_events', 'context', 'implications'],
            'article': ['main_argument', 'stance', 'rhetorical_strategies']
        }
        
        for key in required_keys.get(domain, []):
            if key not in analysis:
                analysis[key] = "Not specified"

        # Step 5: Generate insightful summary
        insightful_summary = generate_insightful_summary_with_refinement(
            original_text=document,
            basic_summary=basic_summary,
            analysis=analysis,
            domain=domain,
            context=context
        )
        logger.info("Insightful summary generated and refined")

        # Step 6: Fact-checking
        trusted_kb = SimpleTrustedKnowledgeBase(context)  # Using context as a simple knowledge base
        fact_checked_summary = fact_check(insightful_summary, trusted_kb)
        logger.info("Summary fact-checked")

        # Step 7: Collect human feedback (optional in automated pipeline)
        # feedback = collect_human_feedback(fact_checked_summary, "Explanation of the summary generation process")

        return {
            "domain": domain,
            "basic_summary": basic_summary,
            "insightful_summary": insightful_summary,
            "fact_checked_summary": fact_checked_summary,
            "analysis": analysis,
            # "feedback": feedback  # Uncomment if using human feedback
        }

    except Exception as e:
        logger.error(f"Error in document summarization: {str(e)}")
        return {"error": str(e)}

def main():
    # Example usage
    document = """
    Climate change is one of the most pressing issues facing our planet today. It refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities, especially the burning of fossil fuels.

    These activities release greenhouse gases into the atmosphere, primarily carbon dioxide and methane. These gases trap heat from the sun, causing the Earth's average temperature to rise. This phenomenon is known as the greenhouse effect.

    The impacts of climate change are far-reaching and significant. They include more frequent and severe weather events, such as hurricanes, droughts, and heatwaves. Rising sea levels threaten coastal communities and islands. Changes in temperature and precipitation patterns affect agriculture and food security.

    Moreover, climate change poses risks to human health, biodiversity, and economic stability. It exacerbates existing social and economic inequalities, as vulnerable populations often bear the brunt of its effects.

    Addressing climate change requires a multi-faceted approach. This includes transitioning to renewable energy sources, improving energy efficiency, protecting and restoring ecosystems, and adapting to the changes already set in motion. International cooperation, as seen in agreements like the Paris Accord, is crucial in coordinating global efforts to mitigate climate change.

    While the challenge is immense, there is still time to act. Every individual, business, and government has a role to play in reducing greenhouse gas emissions and building a sustainable future. The choices we make today will determine the world we leave for future generations.
    """

    result = summarize_document(document)

    if "error" not in result:
        print(f"Document Type: {result['domain']}")
        print("\nBasic Summary:")
        print(result['basic_summary'])
        print("\nInsightful Summary:")
        print(result['insightful_summary'])
        print("\nFact-checked Summary:")
        print(result['fact_checked_summary'])
        print("\nAnalysis:")
        print(result['analysis'])
    else:
        print(f"An error occurred: {result['error']}")

if __name__ == "__main__":
    main()