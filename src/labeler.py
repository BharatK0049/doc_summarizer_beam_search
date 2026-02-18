from rouge_score import rouge_scorer
import numpy as np

class GreedyLabeler:
    def __init__(self):
        # We use ROUGE-L (Longest Common Subsequence) to find the best overlap
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def label_article(self, article_sentences, gold_summary_text, top_k=3):
        """
        Labels the top_k sentences in an article that best represent the gold summary.
        """
        scores = []
        for sent in article_sentences:
            # Score each sentence against the entire gold summary
            score = self.scorer.score(gold_summary_text, sent)['rougeL'].fmeasure
            scores.append(score)
        
        # Identify indices of the top_k scoring sentences
        # We use argsort and take the last top_k indices
        top_indices = np.argsort(scores)[-top_k:]
        
        # Create binary labels: 1 if in top_k, 0 otherwise
        labels = [1 if i in top_indices else 0 for i in range(len(article_sentences))]
        
        return labels

if __name__ == "__main__":
    # Quick Test
    labeler = GreedyLabeler()
    article = [
        "The cat sat on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "The sun is shining brightly today."
    ]
    summary = "A fox jumped over a dog."
    
    labels = labeler.label_article(article, summary, top_k=1)
    print(f"Article Sentences: {article}")
    print(f"Gold Summary: {summary}")
    print(f"Extractive Labels: {labels}") # Expected [0, 1, 0]