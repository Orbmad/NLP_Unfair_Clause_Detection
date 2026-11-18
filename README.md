# Unfair Clause Detection in Terms of Service

## Overview

Online Terms of Service (ToS) agreements often contain complex legal language and clauses that may grant service providers excessive rights at the expense of consumers. Since these agreements are typically lengthy and difficult to interpret, users rarely review them in detail, making it challenging to identify potentially unfair contractual conditions.

This project explores the application of **Natural Language Processing (NLP) and deep learning** to automatically identify potentially unfair clauses in online Terms of Service agreements.

Given the text of an individual contractual clause, the system predicts whether it belongs to one or more of five categories of potential unfairness. The problem is approached as a **multi-label text classification task**, allowing a single clause to be associated with multiple categories or none at all.

The project investigates both traditional neural sequence models and domain-specific transformer architectures, comparing their ability to understand legal language and detect potentially unfair contractual provisions.

## Classification Categories

The system identifies five categories of potentially unfair clauses:

| Label | Category | Description |
|---|---|---|
| `A` | Arbitration | Clauses requiring disputes to be resolved through arbitration rather than traditional courts. |
| `CH` | Unilateral Change | Clauses allowing providers to modify contractual terms or services at their discretion. |
| `CR` | Content Removal | Clauses granting providers the right to remove or restrict user-generated content. |
| `LTD` | Limitation of Liability | Clauses limiting or excluding the provider's liability towards consumers. |
| `TER` | Unilateral Termination | Clauses allowing providers to terminate agreements or user accounts unilaterally. |

Each category is treated as an independent binary prediction. Consequently, the model produces a five-dimensional output vector for every clause.

## Dataset

The project uses the **ToS-100 dataset**, a corpus containing 100 Terms of Service documents divided into 20,417 individual clauses.

The dataset is available through Zenodo:

- [ToS-100 Dataset](https://doi.org/10.5281/zenodo.22691656)

It includes three predefined partitions:

| Split | Documents | Clauses |
|---|---:|---:|
| Training | 80 | 15,837 |
| Validation | 10 | 2,548 |
| Test | 10 | 2,032 |
| **Total** | **100** | **20,417** |

Each clause is annotated with five binary labels corresponding to the classification categories described above.

An important characteristic of this dataset is its significant class imbalance: approximately 92.3% of all clauses do not belong to any of the five unfairness categories.

The original document-based partitions are preserved throughout the experiments to prevent data leakage between training, validation, and test sets.

## Methodology

The project follows a complete NLP classification pipeline, from text preprocessing and word representation to neural network training and evaluation.

### 1. Text Preprocessing

The original corpus contains legal text that has already been tokenized and normalized. Additional preprocessing is applied to improve consistency and prepare the text for neural models.

The preprocessing pipeline includes:

- Restoration of bracket and quotation tokens.
- Normalization of punctuation and whitespace.
- Reattachment of detached contractions and possessives.
- Removal of clause numbering and leading bullets.
- Removal of unnecessary special characters.
- Lemmatization to reduce words to their base forms.

Particular care is taken to preserve semantically important elements of legal language, including modal verbs, negations, and pronouns, which can substantially influence the meaning of contractual clauses.

### 2. Word Embeddings

For the recurrent neural network models, words are represented using pre-trained **GloVe embeddings**.

A vocabulary is constructed to cover all tokens appearing in the training corpus. Words not present in the pre-trained embedding vocabulary receive custom embeddings, while previously unseen words in the validation and test sets are mapped to a special unknown token.

This approach allows the models to handle legal terminology and domain-specific vocabulary that may not be adequately represented by general-purpose embeddings.

### 3. Neural Network Architectures

Two recurrent neural network architectures are investigated.

**Bidirectional LSTM (Baseline)**

The baseline architecture consists of a Bidirectional Long Short-Term Memory (BiLSTM) layer followed by a fully connected output layer.

Processing sequences in both directions allows the model to capture contextual information from preceding and following words.

**Stacked Bidirectional LSTM**

The second architecture extends the baseline by introducing an additional Bidirectional LSTM layer.

The stacked architecture is designed to capture more complex contextual representations and potentially improve the detection of subtle patterns in legal language.

Both architectures produce five independent outputs, one for each unfairness category.

### 4. Transformer-Based Classification

To investigate the benefits of domain-specific pre-training, the project also explores **LEGAL-BERT**, a transformer model pre-trained on legal documents.

The model used is:

- [LEGAL-BERT Base Uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased)

LEGAL-BERT provides contextualized word representations and is specifically designed to capture the linguistic characteristics of legal texts.

The model is fine-tuned on the ToS-100 training set using the Hugging Face Transformers framework.

## Training Strategy

All models are trained using the predefined training partition, with the validation set used for model selection and hyperparameter decisions.

Given the strong imbalance between positive and negative labels, class imbalance is explicitly considered during training.

The evaluation process also considers the possibility of tuning classification thresholds using validation data.

To improve the reliability of experimental results, neural models are trained using multiple random seeds. Performance is summarized using the mean and standard deviation across runs.

## Evaluation

Since this is a multi-label classification problem with highly imbalanced categories, accuracy is not considered an appropriate primary evaluation metric.

Instead, model performance is assessed using:

- **Macro F1-score:** The average F1-score across the five categories, giving equal importance to each category.
- **Macro Precision:** The average precision across the five categories.
- **Macro Recall:** The average recall across the five categories.
- **Per-category F1-score:** Individual F1-scores for Arbitration, Unilateral Change, Content Removal, Limitation of Liability, and Unilateral Termination.

The validation set is used to select the best-performing model, primarily according to macro F1-score. Final evaluation is performed on the held-out test set.

## Technologies

The project uses the following technologies and libraries:

- **Python** – Main programming language.
- **Jupyter Notebook** – Experimental environment and workflow organization.
- **Pandas & NumPy** – Data loading, preprocessing, and numerical operations.
- **TensorFlow / Keras** – Implementation and training of recurrent neural networks.
- **GloVe** – Pre-trained word embeddings.
- **Hugging Face Transformers** – Loading and fine-tuning LEGAL-BERT.
- **Scikit-learn** – Classification metrics and model evaluation.
- **NLTK** – Natural language preprocessing and lemmatization.

## Project Structure

The repository is organized around the main experimental notebook:

```text
.
├── Assignment1-2627_cleanCopy.ipynb
├── README.md
└── data/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

The dataset and pre-trained embeddings are external resources and should be downloaded separately rather than committed to the repository.

## Getting Started

### Prerequisites

- Python 3.10 or later.
- Jupyter Notebook or JupyterLab.
- Access to the ToS-100 dataset.
- Sufficient computational resources to train recurrent neural networks and fine-tune LEGAL-BERT.

A GPU is recommended for faster transformer training.

### Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn
pip install tensorflow scikit-learn nltk
pip install transformers datasets accelerate torch
pip install gensim
```

### Dataset Setup

Download the ToS-100 dataset from [Zenodo](https://doi.org/10.5281/zenodo.22691656).

Extract the three CSV files and place them in the `data/` directory.

The notebook expects the training, validation, and test partitions to be available for loading and preprocessing.

### Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Assignment1-2627_cleanCopy.ipynb` and execute the notebook cells in order.

The notebook contains the complete experimental workflow, including dataset preparation, text preprocessing, embedding construction, model implementation, training, and evaluation.

## Experimental Results

The experiments are designed to compare the effectiveness of recurrent neural networks and domain-specific transformer models for unfair clause detection.

The main comparison considers:

- Baseline Bidirectional LSTM.
- Stacked Bidirectional LSTM.
- Fine-tuned LEGAL-BERT.

Models are compared using macro precision, macro recall, macro F1-score, and per-category F1-score.

Results should be interpreted with particular attention to the rare categories, since their limited number of positive examples makes reliable detection especially challenging.

## Future Improvements

Several directions could further improve the system:

- Investigating alternative class-imbalance mitigation strategies.
- Comparing general-purpose and legal-domain word embeddings.
- Evaluating additional transformer architectures.
- Incorporating neighboring clauses to exploit contractual context.
- Exploring ensemble methods to combine predictions from multiple models.
- Improving classification thresholds for individual unfairness categories.

## References

- Lippi, M. et al. (2019). *CLAUDETTE: an Automated Detector of Potentially Unfair Clauses in Online Terms of Service*. Artificial Intelligence and Law.
- Ruggeri, F. et al. (2022). *Detecting and Explaining Unfairness in Consumer Contracts through Memory Networks*. Artificial Intelligence and Law.
- [ToS-100 Dataset – Zenodo](https://doi.org/10.5281/zenodo.22691656).
- [LEGAL-BERT – Hugging Face](https://huggingface.co/nlpaueb/legal-bert-base-uncased).
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/).

---

**Project focus:** Applying natural language processing and deep learning to improve the automatic identification of potentially unfair contractual clauses in online Terms of Service agreements.
