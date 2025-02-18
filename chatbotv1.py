import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import pdfplumber
from docx import Document
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
import schedule
import time
import threading
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load curated data
with open('curated_data.json', 'r') as f:
    docs = json.load(f)

# Prepare document texts and embeddings
doc_texts = [doc['content'] for doc in docs]
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(doc_texts, convert_to_numpy=True)

# Set up FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Load GPT-Neo model
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Pre-warm the model
model.eval()  # Set the model to evaluation mode
dummy_input = tokenizer("dummy input", return_tensors="pt")
_ = model.generate(dummy_input.input_ids)

# Define retrieval function
def retrieve_documents(query, embedder, index, k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [doc_texts[i] for i in indices[0]]

# Define prompt construction
def construct_prompt(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    return f"\n{context}\n {query}\n"

# Define response generation
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    # Truncate input to a maximum of 512 tokens
    max_input_length = 512
    if inputs.input_ids.shape[1] > max_input_length:
        inputs.input_ids = inputs.input_ids[:, :max_input_length]

    max_new_tokens = 50  # Adjust as needed
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=2,
        early_stopping=True,
        attention_mask=inputs['attention_mask'],  # Pass the attention mask
        pad_token_id=tokenizer.eos_token_id  # Set pad token ID
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define a function to log prompts and responses
def log_data(prompt, response):
    log_entry = {
        "prompt": prompt,
        "response": response
    }
    # Ensure the log file exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # Append the log entry to a JSON file
    with open('logs/collected_data.json', 'a') as log_file:
        log_file.write(json.dumps(log_entry) + "\n")
    print("Collected data has been created and logged.")

# Define chatbot response function
def chatbot_response(query):
    # Pre-defined casual responses
    casual_responses = {
        "greeting": "Hello! How can I help you today?",
        "status": "I'm just a program, but I'm here to help you!",
        "farewell": "Goodbye! It was nice chatting with you.",
        "thanks": "You're welcome! It was my pleasure to assist you.",
        "how are you": "I'm just a program, but I'm functioning properly. How can I help you today?"
    }
    
    normalized_query = query.lower().strip()
    # Use regex to match greetings
    if re.search(r'\b(hi|hello|hey)\b', normalized_query):
        response = casual_responses["greeting"]
    # Use regex to match status inquiry
    elif re.search(r'\b(how are you|what\'s up|status)\b', normalized_query):
        response = casual_responses["status"]
    # Use regex to match farewell
    elif re.search(r'\b(goodbye|bye|farewell)\b', normalized_query):
        response = casual_responses["farewell"]
    # Use regex to match thanks
    elif re.search(r'\b(thank you|thanks)\b', normalized_query):
        response = casual_responses["thanks"]
    # Use regex to match how are you
    elif re.search(r'\b(how are you|how do you do)\b', normalized_query):
        response = casual_responses["how are you"]
    else:
        retrieved_docs = retrieve_documents(query, embedder, index)
        prompt = construct_prompt(query, retrieved_docs)
        response = generate_response(prompt)
    
    # Log the prompt and response
    log_data(query, response)
    return response

app = Flask(__name__)
CORS(app)



@app.route("/", methods=["GET"])
def home():
    return "Hello, world! I'm running."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")
    response = chatbot_response(query)
    return jsonify({"response": response})

@app.route("/check_git", methods=["GET"])
def check_git():
    return jsonify({"current_directory": os.getcwd(), "is_git_repo": os.path.exists('.git')})

@app.route("/list_dir", methods=["GET"])
def list_directory():
    # List the current directory
    dir_contents = subprocess.run(['ls'], capture_output=True, text=True)
    return jsonify({"directory_contents": dir_contents.stdout}), 200
@app.errorhandler(502)
def bad_gateway_error(error):
    return jsonify({"error": "Bad Gateway. Please try again later."}), 502

def ensure_git_repo():
    if not os.path.exists('.git'):
        logging.info("Cloning the repository into a temporary directory...")
        # Clone the repository into a temporary directory
        temp_dir = '/tmp/repo_clone'
        subprocess.run(['git', 'clone', 'https://github.com/lacebx/me0-mini_mvp.git', temp_dir])  # Update the URL as needed
        
        # Copy the necessary files back to the original directory
        for item in os.listdir(temp_dir):
            s = os.path.join(temp_dir, item)
            d = os.path.join('/app', item)  # Assuming '/app' is your working directory
            if os.path.isdir(s):
                if not os.path.exists(d):  # Check if the directory already exists
                    shutil.copytree(s, d, False, None)
                    logging.info(f"Copied directory {s} to {d}.")
            else:
                shutil.copy2(s, d)
                logging.info(f"Copied file {s} to {d}.")

        # Change to the original directory
        os.chdir('/app')  # Change to your working directory
        logging.info("Repository cloned and files copied.")
    else:
        logging.info("Repository already exists. Appending to collected_data.json.")

    # Append new interaction to collected_data.json
    append_to_collected_data()

def append_to_collected_data():
    log_entry = {
        "prompt": "User prompt here",  # Replace with actual user prompt
        "response": "Model response here"  # Replace with actual model response
    }
    
    # Ensure the log file exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Append the log entry to the JSON file
    log_file_path = 'logs/collected_data.json'
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r+') as log_file:
                data = json.load(log_file)
                data.append(log_entry)  # Append new entry
                log_file.seek(0)  # Move the cursor to the beginning of the file
                json.dump(data, log_file, indent=4)  # Write updated data back to the file
                log_file.truncate()  # Remove any leftover data
                logging.info("Appended new interaction to collected_data.json.")
        except json.JSONDecodeError:
            logging.error("JSONDecodeError: The collected_data.json file is corrupted. Creating a new file.")
            with open(log_file_path, 'w') as log_file:
                json.dump([log_entry], log_file, indent=4)  # Create new file with the first entry
                logging.info("Created collected_data.json and added the first interaction.")
    else:
        with open(log_file_path, 'w') as log_file:
            json.dump([log_entry], log_file, indent=4)  # Create new file with the first entry
            logging.info("Created collected_data.json and added the first interaction.")

def push_to_github():
    ensure_git_repo()  # Ensure the repo is cloned and set up

    # Set Git user name and email
    subprocess.run(['git', 'config', '--global', 'user.email', 'your_email@example.com'])  # Replace with your email
    subprocess.run(['git', 'config', '--global', 'user.name', 'Your Name'])  # Replace with your name
    print("GitHub user configuration set.")

    # Get the GitHub token from environment variables
    token = os.environ.get('GITHUB_TOKEN')  # Ensure this is set in Railway environment

    if token:
        # Configure Git to use the token for authentication
        subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'])
        with open(os.path.expanduser('~/.git-credentials'), 'w') as f:
            f.write(f'https://{token}:x-oauth-basic@github.com\n')
        print("GitHub token configured for authentication.")

    # Add the collected_data.json file to git
    subprocess.run(['git', 'add', 'logs/collected_data.json'])  # Adjust the path as necessary
    print("Collected data added to Git staging area.")

    # Commit the changes
    subprocess.run(['git', 'commit', '-m', 'Update collected_data.json'])
    print("Changes committed to Git.")

    # Push to the repository
    push_result = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True, text=True)  # Update 'main' if your branch is different
    if push_result.returncode == 0:
        print("Changes pushed to GitHub successfully.")
    else:
        print(f"Failed to push changes to GitHub: {push_result.stderr}")

# Schedule the job to run every hour (adjust as needed)
schedule.every(.5).hours.do(push_to_github)

# Run the scheduler in a separate thread
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Start the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.start()

if __name__ == "__main__":
    # For production using Gunicorn, the PORT will be managed via Procfile.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
