# classify_document_type.py

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report
import logging
from typing import List, Tuple, Dict
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentClassifier:
    def __init__(self, model_path: str = 'document_classifier_model.joblib', vectorizer_path: str = 'tfidf_vectorizer.joblib'):
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        except FileNotFoundError:
            logger.error(f"Model or vectorizer file not found. Please train the model first.")
            raise
        except Exception as e:
            logger.error(f"Error loading model or vectorizer: {str(e)}")
            raise

        self.label_map: Dict[int, str] = {0: 'scientific_paper', 1: 'news', 2: 'article'}

    def preprocess(self, text: str) -> str:
        # Improved preprocessing: lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def classify(self, document: str) -> str:
        preprocessed_doc = self.preprocess(document)
        vectorized_doc = self.vectorizer.transform([preprocessed_doc])
        prediction = self.model.predict(vectorized_doc)
        return self.label_map[prediction[0]]

def train_document_classifier(train_data: List[str], train_labels: List[int]) -> Tuple[float, str]:
    """
    Train a document classifier using TF-IDF and Naive Bayes.
    
    Args:
    train_data (List[str]): List of document texts
    train_labels (List[int]): List of corresponding labels (0: scientific_paper, 1: news, 2: article)
    
    Returns:
    Tuple[float, str]: (evaluation score, classification_report)
    """
    # Create and fit the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(train_data)

    # Train the Naive Bayes model
    model = MultinomialNB()

    # Determine the appropriate evaluation method based on dataset size
    n_samples = len(train_labels)
    n_classes = len(set(train_labels))
    min_samples_per_class = min(train_labels.count(i) for i in set(train_labels))

    if n_samples >= 30 and min_samples_per_class >= 3:
        # Use cross-validation if we have enough samples
        n_splits = min(5, min_samples_per_class)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_vectorized, train_labels, cv=cv)
        avg_score = np.mean(cv_scores)
        logger.info(f"Using {n_splits}-fold cross-validation")
    else:
        # Fall back to a simple train-test split for very small datasets
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, train_labels, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        avg_score = model.score(X_test, y_test)
        logger.info("Using simple train-test split due to small dataset size")

    # Train on the full dataset
    model.fit(X_vectorized, train_labels)

    # Generate classification report
    y_pred = model.predict(X_vectorized)
    report = classification_report(train_labels, y_pred, target_names=['scientific_paper', 'news', 'article'])

    # Save the model and vectorizer
    joblib.dump(model, 'document_classifier_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

    return avg_score, report

def classify_document_type(document: str) -> str:
    """
    Classify the type of the input document.

    Args:
    document (str): The text of the document to be classified

    Returns:
    str: The predicted document type ('scientific_paper', 'news', 'article', or 'unknown')
    """
    try:
        classifier = DocumentClassifier()
        document_type = classifier.classify(document)
        return document_type
    except Exception as e:
        logger.error(f"An error occurred while classifying the document: {str(e)}")
        return "unknown"

def main():
    # Example of training the model (you would do this with a large dataset)
    logger.info("Training the model with sample data...")
    train_data = [
        "This scientific study explores the effects of climate change on biodiversity.",
        "Breaking news: Major earthquake strikes the coast, thousands evacuated.",
        "In this article, we discuss the importance of mental health awareness.",
        "The researchers conducted a double-blind study to test the efficacy of the new drug.",
        "Tonight's news: Local elections results are in, showing a shift in voter preferences.",
        "This opinion piece argues for stricter regulations on social media platforms."
    ]
    train_labels = [0, 1, 2, 0, 1, 2]  # 0: scientific_paper, 1: news, 2: article
    
    score, report = train_document_classifier(train_data, train_labels)
    logger.info(f"Model trained with evaluation score: {score:.4f}")
    logger.info("Classification Report:\n%s", report)

    # Test the classifier with a sample document
    logger.info("Testing the classifier with a sample document...")
    sample_document = """
    Recent advancements in quantum computing have shown promising results in solving complex optimization problems. 
    This paper presents a novel approach to quantum annealing that demonstrates a significant speedup in finding 
    global optima for NP-hard problems. Our experimental results, conducted on a 2000-qubit system, show...
    """
    
    document_type = classify_document_type(sample_document)
    logger.info(f"The document type is: {document_type}")

if __name__ == "__main__":
    main()