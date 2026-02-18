import pickle
from pathlib import Path
from tqdm import tqdm
from data_loader import load_cnn_data
from beam_search import refined_beam_search
from feature_extractor import SummarizationFeatureExtractor
from labeler import GreedyLabeler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

class MEMSummarizer:
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', DictVectorizer(sparse=True)),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=200, verbose=1))
        ])
        self.fe = SummarizationFeatureExtractor()
        self.labeler = GreedyLabeler()

    def prepare_training_data(self, data):
        X, y = [], []
        print("Extracting features and generating extractive labels...")
        for example in tqdm(data, desc="Processing Articles"):
            article_sents = example['article']
            gold_summary_text = " ".join(example['summary'])
            labels = self.labeler.label_article(article_sents, gold_summary_text)
            doc_stats = self.fe.get_document_stats(article_sents)
            for i, sent in enumerate(article_sents):
                features = self.fe.extract_features(sent, i, doc_stats)
                X.append(features)
                y.append(labels[i])
        return X, y

    def train(self, data):
        X, y = self.prepare_training_data(data)
        print(f"Training MEM on {len(X)} sentence instances...")
        self.model.fit(X, y)

def main():
    train_data = load_cnn_data(split='train', count=500)
    MODEL_FILE = "summarizer_mem.pkl"
    
    if Path(MODEL_FILE).exists():
        print(f"\nLoading saved Summarizer Model from {MODEL_FILE}...")
        with open(MODEL_FILE, 'rb') as f:
            summarizer = pickle.load(f)
    else:
        print("\nTraining new Maximum Entropy Summarizer...")
        summarizer = MEMSummarizer()
        summarizer.train(train_data)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(summarizer, f)

    print("\n" + "="*55)
    print("TESTING REFINED BEAM SEARCH ON MULTIPLE ARTICLES")
    print("="*55)

    boilerplate = ["Editor's note:", "CNN's", "Reuters --", "CNN --"]

    for i in range(0, 4): 
        test_article = train_data[i]['article']
        print(f"\n--- Article {i} ---")
        
        doc_stats = summarizer.fe.get_document_stats(test_article)
        X_test = [summarizer.fe.extract_features(s, idx, doc_stats) for idx, s in enumerate(test_article)]
        probs = summarizer.model.predict_proba(X_test)[:, 1]
        
        # TRANSFORMATION STEP: Simple String Cleaning
        cleaned_sentences = []
        for sent in test_article:
            new_sent = sent
            for b in boilerplate:
                new_sent = new_sent.replace(b, "").strip()
            cleaned_sentences.append(new_sent)
        
        print(f"-> Transformation Model applied: Stripped metadata and attributions.")

        # PROBABILISTIC STEP: Beam Search with Redundancy Penalty
        final_summary = refined_beam_search(
            probs, 
            cleaned_sentences, 
            beam_width=5, 
            summary_length=3, 
            redundancy_penalty=0.9
        )

        print("Generated Summary:")
        for sent in final_summary:
            print(f"- {sent}")

if __name__ == "__main__":
    main()