import pandas as pd

# Function to extract English and Oshikwanyama sentences and remove quotation marks
def extract_sentences(text):
    # Split text by lines
    lines = text.split('\n')
    
    # Initialize lists to store English and Oshikwanyama sentences
    english_sentences = []
    oshikwanyama_sentences = []

    # Iterate through each line
    for line in lines:
        # Remove numbers at the beginning of each row
        line_parts = line.split(",")[1:]  # Skip the first element (numbers)
        line = ",".join(line_parts)  # Join the remaining parts
        # Split line by comma
        parts = line.split("..,")
        if len(parts) >= 2:
            # Remove quotation marks from English sentence and add to list
            english_sentences.append(parts[0].strip().replace('"', ''))
            # Remove quotation marks from Oshikwanyama sentence and add to list
            oshikwanyama_sentences.append(parts[1].strip().replace('"', ''))

    return english_sentences, oshikwanyama_sentences


# Main function
def main():
    # Read the provided text
    with open("genesis_sentences.csv", "r", encoding="utf-8") as file:
        text = file.read()

    # Extract English and Oshikwanyama sentences
    english_sentences, oshikwanyama_sentences = extract_sentences(text)

    # Create DataFrames to store the sentences
    df_english = pd.DataFrame({"English": english_sentences})
    df_oshikwanyama = pd.DataFrame({"Oshikwanyama": oshikwanyama_sentences})

    # Write DataFrames to separate CSV files
    df_english.to_csv("small_vocab_en.csv", index=False)
    df_oshikwanyama.to_csv("small_vocab_osh.csv", index=False)

    print("CSV files created successfully.")

if __name__ == "__main__":
    main()
