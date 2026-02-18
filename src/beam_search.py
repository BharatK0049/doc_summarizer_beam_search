import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def beam_search_decoder(probs, sentences, beam_width=3, summary_length=3):
    """
    Finds the most probable sequence of sentences for the summary.
    
    Args:
        probs (list): Probability of each sentence being "summary-worthy".
        sentences (list): The actual sentence strings.
        beam_width (int): Number of sequences to keep at each step.
        summary_length (int): Total sentences to include in the summary.
    """
    # Each item in the beam: (total_log_prob, [list_of_indices])
    # We start with an empty summary
    beam = [(0.0, [])]
    
    for _ in range(summary_length):
        candidates = []
        for score, current_seq in beam:
            for i in range(len(probs)):
                # Avoid picking the same sentence twice
                if i not in current_seq:
                    # We add the log probability to avoid underflow
                    # score + log(P(sentence_i is summary))
                    new_score = score + np.log(probs[i] + 1e-10) 
                    new_seq = current_seq + [i]
                    candidates.append((new_score, new_seq))
        
        # Sort by score and keep only the top 'beam_width' sequences
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]
    
    # Return the best sequence of sentences
    best_indices = beam[0][1]
    return [sentences[idx] for idx in sorted(best_indices)]

def refined_beam_search(probs, sentences, beam_width=3, summary_length=3, redundancy_penalty=0.5):
    vectorizer = CountVectorizer().fit(sentences)
    beam = [(0.0, [])]
    
    for _ in range(summary_length):
        candidates = []
        for score, current_seq in beam:
            # Vectorize the existing summary to check for similarity
            summary_vec = None
            if current_seq:
                summary_text = " ".join([sentences[idx] for idx in current_seq])
                summary_vec = vectorizer.transform([summary_text])

            for i in range(len(probs)):
                if i not in current_seq:
                    # Probabilistic Refinement: Penalty for redundancy
                    current_penalty = 0
                    if summary_vec is not None:
                        sent_vec = vectorizer.transform([sentences[i]])
                        similarity = cosine_similarity(summary_vec, sent_vec)[0][0]
                        current_penalty = similarity * redundancy_penalty
                    
                    # New Score = Log Prob - Redundancy Penalty
                    new_score = score + np.log(probs[i] + 1e-10) - current_penalty
                    candidates.append((new_score, current_seq + [i]))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]
    
    return [sentences[idx] for idx in sorted(beam[0][1])]