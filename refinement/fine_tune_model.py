import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load your cleaned dataset
dataset = load_dataset('json', data_files='cleaned_data.json')

# Split the dataset into train and test sets
train_test_split = dataset['train'].train_test_split(test_size=0.2)  # 20% for testing

# Load the tokenizer and model
model_name = "EleutherAI/gpt-neo-125M"  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to be the same as the EOS token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['prompt'], padding="max_length", truncation=True)

tokenized_datasets = train_test_split.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create a Trainer instance
model = AutoModelForCausalLM.from_pretrained(model_name)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")