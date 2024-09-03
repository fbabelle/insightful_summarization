import datetime
import json
import os
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackDatabase:
    def __init__(self, filename: str = 'feedback_database.json'):
        self.filename = filename
        self.feedback_data: List[Dict[str, Any]] = self.load_feedback()

    def load_feedback(self) -> List[Dict[str, Any]]:
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    return json.load(f)
            return []
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.filename}. Starting with empty feedback data.")
            return []
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}")
            return []

    def save_feedback(self) -> None:
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback data: {str(e)}")

    def add_feedback(self, feedback: Dict[str, Any]) -> None:
        self.feedback_data.append(feedback)
        self.save_feedback()

    def get_all_feedbacks(self) -> List[Dict[str, Any]]:
        return self.feedback_data

def get_rating(prompt: str, min_value: int = 1, max_value: int = 5) -> int:
    """Helper function to get a valid rating from the user."""
    while True:
        try:
            rating = int(input(prompt))
            if min_value <= rating <= max_value:
                return rating
            else:
                logger.warning(f"Please enter a number between {min_value} and {max_value}.")
        except ValueError:
            logger.warning("Please enter a valid number.")

def get_text_input(prompt: str) -> str:
    """Helper function to get non-empty text input from the user."""
    while True:
        response = input(prompt).strip()
        if response:
            return response
        logger.warning("Please enter a non-empty response.")

def collect_human_feedback(summary: str, explanation: str) -> Dict[str, Any]:
    """
    Collects human feedback on the generated summary and explanation.

    Args:
    summary (str): The generated summary
    explanation (str): The explanation of how the summary was generated

    Returns:
    Dict[str, Any]: A dictionary containing the collected feedback
    """
    logger.info("\nPlease provide feedback on the generated summary and explanation:\n")
    logger.info("Summary:\n%s", summary)
    logger.info("\nExplanation:\n%s", explanation)

    # Collect ratings
    ratings = {
        "clarity": get_rating("How clear is the summary? (1-5): "),
        "accuracy": get_rating("How accurate is the summary? (1-5): "),
        "completeness": get_rating("How complete is the summary? (1-5): "),
        "explanation_helpfulness": get_rating("How helpful is the explanation? (1-5): ")
    }

    # Collect text feedback
    text_feedback = {
        "strengths": get_text_input("What are the strengths of this summary? "),
        "weaknesses": get_text_input("What are the weaknesses or areas for improvement? "),
        "additional_comments": get_text_input("Any additional comments? ")
    }

    # Compile feedback
    feedback = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": summary,
        "explanation": explanation,
        "ratings": ratings,
        "text_feedback": text_feedback
    }

    # Store feedback
    db = FeedbackDatabase()
    db.add_feedback(feedback)

    logger.info("Thank you for your feedback!")

    return feedback

def export_all_feedbacks(output_file: str = '/home/dfoadmin/boqu/insightful_summarization/user_feedback/all_feedbacks.json') -> None:
    """
    Retrieves all feedbacks and stores them in a JSON file.

    Args:
    output_file (str): The name of the file to store the feedbacks in.
    """
    db = FeedbackDatabase()
    all_feedbacks = db.get_all_feedbacks()
    
    try:
        with open(output_file, 'w') as f:
            json.dump(all_feedbacks, f, indent=2)
        logger.info(f"All feedbacks have been exported to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting feedbacks: {str(e)}")

def main():
    # Collect some sample feedbacks
    for i in range(3):
        sample_summary = f"This is sample summary {i+1}."
        sample_explanation = f"This is explanation for summary {i+1}."
        collect_human_feedback(sample_summary, sample_explanation)

    # Export all feedbacks
    export_all_feedbacks()

    # Display the exported feedbacks
    with open('all_feedbacks.json', 'r') as f:
        exported_feedbacks = json.load(f)
    
    logger.info("\nExported Feedbacks:")
    logger.info(json.dumps(exported_feedbacks, indent=2))

if __name__ == "__main__":
    main()