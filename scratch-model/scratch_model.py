import pandas as pd
from datasets import Dataset
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load the data from a CSV file
df = pd.read_csv("engoshi.csv")

# Split the data into train and validation sets
train_test_split = df.sample(frac=0.8, random_state=42)
train_dataset = Dataset.from_pandas(train_test_split)
valid_dataset = Dataset.from_pandas(df.drop(train_test_split.index))

# Load a pretrained MarianMT model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-mul-tiny"  # Using a smaller model
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Preprocess the data
def preprocess_function(examples):
    model_inputs = tokenizer(examples["english"], max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["oshikwanyama"], max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
valid_dataset = valid_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reduce batch size further
    per_device_eval_batch_size=2,   # Reduce batch size further
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)
