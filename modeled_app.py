import pandas as pd
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.translate import AlignedSent, IBMModel2

app = Flask(__name__)

nltk.download('punkt')

# Load the parallel corpus from a CSV file
def load_parallel_corpus(file_path):
    df = pd.read_csv(file_path)
    print("Columns in the DataFrame:", df.columns)  # Print column names
    df.dropna(subset=['english', 'oshikwanyama'], inplace=True)  # Drop rows with NaN values
    english_sentences = df['english'].astype(str).tolist()
    oshikwanyama_sentences = df['oshikwanyama'].astype(str).tolist()
    return english_sentences, oshikwanyama_sentences

# Clean sentences by stripping whitespace
def clean_sentences(sentences):
    return [str(sentence).strip() for sentence in sentences]

# Prepare the aligned sentences for training the IBM Model
def prepare_aligned_sentences(english_sentences, oshikwanyama_sentences):
    aligned_sentences = []
    for eng_sent, osh_sent in zip(english_sentences, oshikwanyama_sentences):
        eng_tokens = nltk.word_tokenize(eng_sent.lower())
        osh_tokens = nltk.word_tokenize(osh_sent.lower())
        aligned_sentences.append(AlignedSent(eng_tokens, osh_tokens))
    return aligned_sentences

# Train the IBM Model 2 using the aligned sentences
def train_ibm_model(aligned_sentences, num_iterations=10):
    ibm2 = IBMModel2(aligned_sentences, num_iterations)
    return ibm2

# Translate an English sentence to Oshikwanyama using the trained IBM Model 2
def translate_sentence(ibm2, english_sentence):
    eng_tokens = nltk.word_tokenize(english_sentence.lower())
    translation = []
    for eng_token in eng_tokens:
        if eng_token in ibm2.translation_table:
            best_osh_token = max(ibm2.translation_table[eng_token], key=ibm2.translation_table[eng_token].get)
            translation.append(best_osh_token)
        else:
            translation.append(eng_token)  # Default to English word if no translation found
    return ' '.join(translation)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    english_input = request.form['english_input']
    oshikwanyama_translation = translate_sentence(ibm2, english_input)
    return jsonify({'translation': oshikwanyama_translation})

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"status": 404, "message": "Not Found"}), 404

if __name__ == '__main__':
    file_path = 'engoshi.csv'  # Define your CSV file path here
    english_sentences, oshikwanyama_sentences = load_parallel_corpus(file_path)
    english_sentences = clean_sentences(english_sentences)
    oshikwanyama_sentences = clean_sentences(oshikwanyama_sentences)
    aligned_sentences = prepare_aligned_sentences(english_sentences, oshikwanyama_sentences)
    ibm2 = train_ibm_model(aligned_sentences)
    app.run(debug=True)
