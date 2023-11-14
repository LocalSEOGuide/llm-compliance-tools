import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from difflib import SequenceMatcher
from nltk.sentiment import SentimentIntensityAnalyzer

def is_ai_generated_text(text):
    flags = []

    # Check for repetition
    repetition_pattern = re.compile(r'(\b\w+\b)\s+\1')
    if repetition_pattern.search(text):
        flags.append("Repetitive content")

    # personalization
    personalization_keywords = ['I', 'me', 'my']
    if not any(keyword in text for keyword in personalization_keywords):
        flags.append("Lack of personalization")

    # Sentiment analysis
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_score = sentiment_analyzer.polarity_scores(text)['compound']
    if sentiment_score < 0.1:  # Adjust the threshold as needed
      if sentiment_score > -0.1:
        flags.append("Banality/neutral sentiment")
    # Check for tell-tale phrases
    telltale_phrases = ['Sure,', 'as an AI assistant', 'hidden gem', "Now I know what you're thinking"]
    if any(phrase in text for phrase in telltale_phrases):
        flags.append("tell tale phrase")

        # Check for repetitive structures
    repetitive_structures = ['If you', 'In conclusion', 'In my opinion', 'The fact is']
    if any(structure in text for structure in repetitive_structures):
      flags.append("Repetitive Structures")

    # Check for overuse of generic responses
    generic_responses = ['I understand', 'Thank you for your question', 'Please let me know if you need further assistance']
    if sum(text.count(response) for response in generic_responses) > 1:  # Adjust the threshold as needed
        flags.append("Overuse of generic responses")

    # Chunk the text into sentences
    sentences = sent_tokenize(text)

    # Check for repetitive sentence structures
    structure_similarity_threshold = 0.5  # Adjust this threshold as needed
    if has_repetitive_structure(sentences, structure_similarity_threshold):
        flags.append("Repetitive sentence structures")

    # Check for unnatural wording
    unnatural_pattern = re.compile(r'\b(?:\w+ing|to\s+\w+|an\s+\w+|the\s+\w+|for\s+\w+)\b')
    if unnatural_pattern.search(text):
        flags.append("Unnatural patterns in text")

    # Check for excessive word count
    word_count_threshold = 1000  # Adjust this threshold as needed
    if len(text.split()) > word_count_threshold:
        flags.append("Word count doesn't match expectation")

    # Check for proximity repetition
    proximity_repetition_threshold = 10  # Adjust this threshold as needed
    proximity_similarity_threshold = 0.9  # Adjust this threshold as needed
    if has_proximity_repetition(sentences, proximity_repetition_threshold, proximity_similarity_threshold):
        flags.append("Proximity repetition")

    return flags

def has_proximity_repetition(sentences, proximity_threshold, proximity_similarity_threshold):
    for i in range(len(sentences)):
        for j in range(i+1, min(i+proximity_threshold+1, len(sentences))):
            similarity_score = compute_sentence_similarity(sentences[i], sentences[j])
            if similarity_score >= proximity_similarity_threshold:
                return True
    return False

def has_repetitive_structure(sentences, similarity_threshold):
    # Check for repetitive sentence structures
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            similarity_score = compute_sentence_similarity(sentences[i], sentences[j])
            if similarity_score >= similarity_threshold:
                return True
    return False

def compute_sentence_similarity(sentence1, sentence2):
    # Tokenize the sentences into words
    words1 = word_tokenize(sentence1.lower())
    words2 = word_tokenize(sentence2.lower())

    # Compute word similarity scores using WordNet
    similarity_scores = []
    for word1 in words1:
        for word2 in words2:
            word1_synsets = wn.synsets(word1)
            word2_synsets = wn.synsets(word2)
            if word1_synsets and word2_synsets:
                similarity = max(s1.path_similarity(s2) for s1 in word1_synsets for s2 in word2_synsets)
                similarity_scores.append(similarity)

    # Compute average similarity score
    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
    else:
        avg_similarity = 0

    return avg_similarity

text = "Sure, I can help you with that as an AI assistant. Now I know what you're thinking, it's a hidden gem!"
flags = is_ai_generated_text(text)

if flags:
    print("AI-generated text detected. Flags:")
    for flag in flags:
        print("- " + flag)
else:
    print("Human-generated text.")
