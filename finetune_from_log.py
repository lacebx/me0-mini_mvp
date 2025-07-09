import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

MODEL_NAME = "EleutherAI/gpt-neo-125M"
DATA_FILE = "logs/finetune_pairs.jsonl"
OUTPUT_DIR = "./fine_tuned_model"

# Load data
with open(DATA_FILE, 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

prompts = [d['prompt'] for d in data]
responses = [d['response'] for d in data]

# Prepare dataset
class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, responses, tokenizer, max_length=256):
        self.inputs = [f"User: {p}\nAssistant: {r}" for p, r in zip(prompts, responses)]
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.inputs[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc['labels'] = enc['input_ids'].clone()
        return enc

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

dataset = ChatDataset(prompts, responses, tokenizer)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print(f"Starting fine-tuning on {len(dataset)} pairs...")
trainer.train()
print("Saving fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}") 