import pandas as pd
import nltk

# nltk.data.path.append('/Users/Matheus/nltk_data')


from nltk.translate import AlignedSent, IBMModel2 


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

# Main function to load the corpus, train the model, and translate sentences
def main(file_path):
    english_sentences, oshikwanyama_sentences = load_parallel_corpus(file_path)
    english_sentences = clean_sentences(english_sentences)
    oshikwanyama_sentences = clean_sentences(oshikwanyama_sentences)
    aligned_sentences = prepare_aligned_sentences(english_sentences, oshikwanyama_sentences)
    ibm2 = train_ibm_model(aligned_sentences)

    while True:
        english_input = input("Enter an English sentence to translate (or 'exit' to quit): ")
        if english_input.lower() == 'exit':
            break
        oshikwanyama_translation = translate_sentence(ibm2, english_input)
        print(f"Oshikwanyama Translation: {oshikwanyama_translation}")

# Example usage:
# main('engoshi.csv')

if __name__ == '__main__':
    main('engoshi.csv')
