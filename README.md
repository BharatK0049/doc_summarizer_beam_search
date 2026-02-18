# Document Summarization using Maximum Entropy Model & Beam Search

## Overview

This project implements an **Extractive Document Summarizer** using a **Maximum Entropy Model (MEM)** trained on the CNN/DailyMail dataset.

The system follows a multi-stage pipeline to extract the most informative and non-redundant sentences from a news article and generate a concise **3-sentence summary**.

---

## System Architecture

### 1️⃣ Maximum Entropy Model (MEM)

The core of the summarizer is a **discriminative model** that estimates:

\[
P(y \mid x)
\]

where each sentence is scored based on its *summary-worthiness*.

Unlike word-level models, this system uses **sentence-level feature functions** \( f(x, y) \) to capture structural and contextual signals.

#### Feature Categories

- **Structural Features**
  - `is_lead` (first sentence)
  - `is_second`
  - `relative_pos`  
  Captures the *inverted pyramid* structure common in journalism.

- **Information Density**
  - `num_density`
  - `has_numbers`  
  Detects factual anchors such as dates, ages, statistics, and monetary values.

- **Morphological Hints**
  - `cap_word_count`  
  Proxy for named entity density.

- **Lexical Centrality**
  - `keyword_overlap`  
  Measures overlap with the document’s top-10 most frequent words.

---

### 2️⃣ Sequence Decoding: Beam Search

Rather than selecting top-K sentences independently, the system applies **Beam Search (width = 5)** to find the optimal sentence sequence.

This allows global optimization of the summary instead of greedy local decisions.

#### Redundancy Penalty (Probabilistic Refinement)

During decoding:

- Cosine similarity is computed between candidate sentences and the current summary.
- Redundant sentences receive a **log-probability penalty**.
- This encourages information diversity and prevents repetition.

---

### 3️⃣ Transformation-Based Refinement

A rule-based transformation layer improves readability.

#### Boilerplate Removal

The system automatically strips metadata commonly found in news corpora, such as:

- `Editor's note:`
- `CNN --`
- `Reuters --`

This ensures the final output is clean and natural.

---

## Results and Observations

- **Lead Bias**  
  The MEM correctly learns that important information is concentrated in the first two sentences of news articles.

- **Sequence Optimization**  
  Beam Search reduces redundancy and avoids repetitive sentence selection.

- **Model Persistence**  
  The trained model is serialized using `pickle`, allowing instant inference without retraining.

---

## Technical Components

- **MEM Classifier**  
  Discriminative model using structural and lexical features.

- **Beam Search Decoder**  
  Width = 5 sequence optimization.

- **Redundancy Penalty**  
  Cosine similarity-based diversity control.

- **Transformation Layer**  
  Rule-based metadata cleaning.

---

## Project Structure

```text
doc_summarizer_beam_search/
├── src/
│   ├── data_loader.py       # Fetches and tokenizes CNN/DailyMail dataset
│   ├── feature_extractor.py # Custom sentence-level feature extraction
│   ├── labeler.py           # Greedy ROUGE-L labeler for extractive training
│   ├── beam_search.py       # Sequence decoding with redundancy penalty
│   └── main.py              # Training loop, persistence, and evaluation
├── summarizer_mem.pkl       # Serialized MEM model
├── requirements.txt
└── README.md
```


---

## Installation

### 1. Setup Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Usage

```bash
python src/main.py
```


