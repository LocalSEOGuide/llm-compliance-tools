import spacy
from spacy.matcher import Matcher
from collections import defaultdict


class FactChecker:
    def __init__(self, fact_list):
        self.fact_list = fact_list
        self.nlp = spacy.load('en_core_web_md')  # Changed to medium model
        self.matcher = Matcher(self.nlp.vocab)

    def extract_entities(self, text):
        """Extract named entities and other relevant facts from text"""
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities

    def compute_similarity(self, list1, list2):
        """Compute similarity between two lists using word embeddings"""
        print("List1:", list1)  # Debug print
        print("List2:", list2)  # Debug print

        doc1 = self.nlp(" ".join(list1))
        doc2 = self.nlp(" ".join(list2))

        if doc1.vector_norm == 0 or doc2.vector_norm == 0:
            print("Warning: One of the documents has empty vectors.")
            return 0

        return doc1.similarity(doc2)


    def check_factual_accuracy(self, text):
        """Check the factual accuracy of the text against the fact list"""
        extracted_entities = self.extract_entities(text)
        similarity_score = self.compute_similarity(extracted_entities, self.fact_list)
        return similarity_score >= 0.5

    def log_feedback(self, text, feedback):
        """Log user feedback for future improvements"""
        # This method should ideally log the feedback to a file or database
        print(f"Feedback logged for text: {text}\nFeedback: {feedback}")

# Example usage
fact_checker = FactChecker(["Washington", "2021", "climate change"])
text = "In 2021, the president of the United States addressed climate change."
accuracy = fact_checker.check_factual_accuracy(text)
print(f"Is the text factually accurate? {accuracy}")
