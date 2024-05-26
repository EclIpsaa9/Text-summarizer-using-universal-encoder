# Automatic Text Summarization using Universal Sentence Encoder

This project utilizes the Universal Sentence Encoder to automatically generate summaries of text documents. The process involves several steps including preprocessing, spell checking, embedding sentences, calculating similarity scores, and generating the final summary.

Dependencies
Python 3.x
Libraries: NLTK, NumPy, NetworkX, Tensorflow, SpellChecker, Pandas, Scikit-learn
Pre-trained Universal Sentence Encoder (downloaded during runtime)
Installation
Clone the repository or download the source code.
Install the required dependencies listed in requirements.txt.
Copy code
pip install -r requirements.txt
Usage
Ensure your text data is in a suitable format.
Import the necessary modules and functions from the script.
Prepare your text data and call the appropriate functions for preprocessing, summarization, and evaluation.
Customize parameters such as the number of sentences in the summary (top_n) according to your requirements.
Example
python
Copy code
# Import necessary modules and functions
import nltk
from spellchecker import SpellChecker
import tensorflow_hub as hub
from your_script import read_article, correct_sentence, sentence_similarity, build_similarity_matrix, generate_summary

# Read and preprocess your text data
article = "Your text data here"
sentences = read_article(article)

# Correct spelling mistakes in sentences
corrected_sentences = [correct_sentence(sentence) for sentence in sentences]

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Generate summary
summary = generate_summary(" ".join(corrected_sentences), top_n=5, embeds=embed)
print("Generated Summary:", summary)
License
MIT License

Acknowledgments
The Universal Sentence Encoder: https://tfhub.dev/google/universal-sentence-encoder/4
