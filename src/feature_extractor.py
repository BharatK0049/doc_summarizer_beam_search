import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

class SummarizationFeatureExtractor:
    def __init__(self):
        # We'll use this to calculate how "central" a sentence is to the document
        self.tfidf = TfidfVectorizer(stop_words='english')

    def get_document_stats(self, article_sentences):
        """Pre-calculates document-wide data for relative feature scoring."""
        doc_text = " ".join(article_sentences)
        # Fit TF-IDF on this specific document to find key words
        try:
            self.tfidf.fit(article_sentences)
            self.feature_names = self.tfidf.get_feature_names_out()
        except:
            self.feature_names = []
        
        return {
            'doc_len': len(article_sentences),
            'doc_word_counts': Counter(re.findall(r'\w+', doc_text.lower()))
        }

    def extract_features(self, sentence, index, doc_stats):
        """
        Extracts custom features for a single sentence.
        """
        words = re.findall(r'\w+', sentence.lower())
        sentence_len = len(words)
        
        # 1. Position Feature (Standard for news: earlier is usually better)
        pos_ratio = index / doc_stats['doc_len'] if doc_stats['doc_len'] > 0 else 0
        
        # 2. Length Features
        # Avoid very short sentences (often noise/headers)
        is_short = sentence_len < 5
        
        # 3. Numerical Density (Factual density often implies importance)
        has_numbers = any(char.isdigit() for char in sentence)
        num_count = sum(1 for w in words if w.isdigit())

        # 4. Uppercase/Proper Noun Hint (Presence of Named Entities)
        # Using a simple proxy: check for capitalized words not at the start
        cap_words = sum(1 for i, w in enumerate(sentence.split()) 
                        if i > 0 and w[0].isupper())

        # 5. Centroid Similarity (Keyword Overlap)
        # How many of the top words in the sentence are also frequent in the doc?
        important_word_count = 0
        if doc_stats['doc_word_counts']:
            top_doc_words = [w for w, c in doc_stats['doc_word_counts'].most_common(10)]
            important_word_count = sum(1 for w in words if w in top_doc_words)

        features = {
            'bias': 1.0,
            'is_lead': index == 0,           # First sentence is highly likely a summary
            'is_second': index == 1,
            'relative_pos': round(pos_ratio, 1), 
            'is_short': is_short,
            'num_density': num_count / sentence_len if sentence_len > 0 else 0,
            'has_numbers': has_numbers,
            'cap_word_count': cap_words,
            'keyword_overlap': important_word_count,
            'length_bucket': min(sentence_len // 10, 5), # Bucket lengths (0-10, 10-20, etc.)
        }
        
        return features