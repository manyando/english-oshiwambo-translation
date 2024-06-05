import nltk
import certifi
import os
from tkinter import Tk, Text, Button, END, Label, Scrollbar, RIGHT, Y, LEFT, BOTH

# Ensure SSL certificates are handled properly
os.environ['SSL_CERT_FILE'] = certifi.where()

# Ensure the punkt tokenizer model is downloaded
nltk.download('punkt')

from nltk.tokenize import word_tokenize

def tokenize_text():
    input_text = text_input.get("1.0", END).strip()
    if input_text:
        tokens = word_tokenize(input_text)
        text_output.delete("1.0", END)
        text_output.insert(END, ' '.join(tokens))

# Create the main application window
root = Tk()
root.title("Text Tokenizer")

# Create a Label for instructions
label = Label(root, text="Enter text to tokenize:")
label.pack()

# Create a Text widget for text input
text_input = Text(root, height=10, width=50)
text_input.pack()

# Create a Button to trigger tokenization
tokenize_button = Button(root, text="Tokenize", command=tokenize_text)
tokenize_button.pack()

# Create a Text widget for text output
text_output = Text(root, height=10, width=50)
text_output.pack()

# Add a Scrollbar to the output Text widget
scrollbar = Scrollbar(root, command=text_output.yview)
text_output.config(yscrollcommand=scrollbar.set)
scrollbar.pack(side=RIGHT, fill=Y)

# Run the application
root.mainloop()
