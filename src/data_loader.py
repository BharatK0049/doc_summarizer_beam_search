from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize

def load_cnn_data(split='train', count=1000):
    """
    Loads a subset of the CNN/DailyMail dataset.
    
    Args:
        split (str): 'train', 'validation', or 'test'.
        count (int): Number of articles to load (to save time initially).
    """
    nltk.download('punkt')
    nltk.download('punkt_tab')
    print(f"Downloading CNN/DailyMail dataset ({split})...")
    # version 3.0.0 is the standard for summarization
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"{split}[:{count}]")
    
    # Preprocessing: Tokenize articles into sentences
    processed_data = [] 
    for example in dataset:
        article_sentences = sent_tokenize(example['article'])
        summary_sentences = sent_tokenize(example['highlights'])
        processed_data.append({
            'article': article_sentences,
            'summary': summary_sentences
        })
        
    print(f"Loaded {len(processed_data)} articles from {split} set.")
    return processed_data

if __name__ == "__main__":
    # Test run
    nltk.download('punkt')
    data = load_cnn_data(count=5)
    print(f"Example Article Sentence 1: {data[0]['article'][0]}")
    print(f"Example Summary Sentence 1: {data[0]['summary'][0]}")