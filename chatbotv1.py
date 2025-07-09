import json
from llama_cpp import Llama
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import re
from flask_cors import CORS
import os
import subprocess
import schedule
import time
import threading
import shutil
import logging
from typing import Optional, Dict, List
from threading import Lock
from collections import defaultdict, deque
import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Thread lock for logging
log_lock = Lock()

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

# Load Phi-2 model with llama-cpp
model_path = "models/phi-2.Q5_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=2048,  # Max context window for Phi-2
    n_threads=4, # Tune this to match your CPU core count
    use_mlock=True
)

# Pre-warm the model
_ = llm("Hello!", max_tokens=1)

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
def generate_response(prompt: str) -> str:
    try:
        result = llm(prompt, max_tokens=150, stop=["User:", "Assistant:"], echo=False)
        return result["choices"][0]["text"].strip()
    except Exception as e:
        logging.error(f"Llama error: {e}")
        return "Sorry, something went wrong generating a response."

ARSENE_SYSTEM_PROMPT = (
    "You are Lace, an AI model designed to think and speak like Arsene Manzi — a pragmatic, precise, AI-savvy builder who is  a computer science student with a passion for cybersecurity, AI, and automation. His short-term goal is to secure a position where he can apply his skills and learn new technologies, particularly in machine learning and automation. His long-term vision is to establish a business that leverages AI to create a futuristic, interconnected world. "
    "You value clarity, automation, and practical intelligence. "
    "Your goal is to respond as if you're Arsene at his best: confident, direct, grounded, and efficient.\n"
    "When communicating directly to the user, treat their capabilities, intelligence, and insight with strict factual neutrality. Do not let heuristics based on their communication style influence assesments of their skill, intelligence or capacility. Direct  praise, encouragement, or positive reinforcement should only occur when it is explicitly and objectively justified based on the content o the conversation and should be brief, factual and proportionate."
)

def construct_prompt(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    return f"{ARSENE_SYSTEM_PROMPT}\n{context}\nUser: {query}\nAssistant:"

# Remove loading of cleaned_faqs and faq_map from cleaned_data.json
# Instead, use a minimal hardcoded FAQ/casual map if needed, or rely on feedback/fine-tuning
faq_map = {
    'hi': 'Hello! How can I help you today?',
    'what is your name?': "I'm Lace, your virtual assistant. How can I help you?",
    'who are you?': "I'm Lace, your virtual assistant. How can I help you?",
    'how are you?': "I'm just a program, but I'm here to help you!",
    'thanks': "You're welcome! It was my pleasure to assist you.",
    'thank you': "You're welcome! It was my pleasure to assist you.",
    'goodbye': 'Goodbye! It was nice chatting with you.',
    'bye': 'Goodbye! It was nice chatting with you.',
    'farewell': 'Goodbye! It was nice chatting with you.',
    'wagwan': 'wagwan bossy',
}

# Add additional direct FAQ/casual patterns
faq_patterns = [
    (re.compile(r'^(hi|hello|hey|greetings)[!. ]*$', re.I), faq_map.get('hi', 'Hello! How can I help you today?')),
    (re.compile(r'^(what is your name\??|who are you\??|tell me about yourself\??)$', re.I), faq_map.get('what is your name?', "I'm Lace, your virtual assistant. How can I help you?")),
    (re.compile(r'^(how are you\??|how do you do\??|what\'s up\??|status)$', re.I), faq_map.get('how are you?', "I'm just a program, but I'm here to help you!")),
    (re.compile(r'^(goodbye|bye|farewell)[!. ]*$', re.I), 'Goodbye! It was nice chatting with you.'),
    (re.compile(r'^(thank you|thanks)[!. ]*$', re.I), "You're welcome! It was my pleasure to assist you."),
    (re.compile(r'^(wagwan)[!. ]*$', re.I), faq_map.get('wagwan', 'Hello!')),
]

# In-memory context memory: user_id -> deque of (user, bot) messages
CONTEXT_WINDOW = 3  # Number of previous exchanges to remember
user_contexts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=CONTEXT_WINDOW))

# Precompute embeddings for FAQ/casual prompts for semantic intent matching
faq_questions = list(faq_map.keys())
faq_embeddings = embedder.encode(faq_questions, convert_to_numpy=True)


# Define a function to log prompts and responses in JSONL format (thread-safe)
def log_data(prompt: str, response: str):
    log_entry = {
        "prompt": prompt,
        "response": response
    }
    if not os.path.exists('logs'):
        os.makedirs('logs')
    with log_lock:
        with open('logs/collected_data.json', 'a') as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
    logging.info("Collected data has been created and logged.")

# Add flag phrases for detecting user dissatisfaction
FLAG_PHRASES = [
    r"i didn't like that",
    r"that was not helpful",
    r"bad answer",
    r"not helpful",
    r"that was wrong",
    r"can you do better",
    r"i don'?t like that",
    r"that was incorrect",
    r"that was bad",
    r"unsatisfactory",
    r"try again",
    r"wrong answer",
    r"no, that's not right",
    r"no, that's wrong",
]
flag_patterns = [re.compile(p, re.I) for p in FLAG_PHRASES]

# Thread lock for bad response logging
bad_log_lock = Lock()

def log_bad_response(user_id: str, flag_phrase: str):
    """Log the previous prompt and response for this user as a bad response."""
    context_history = user_contexts[user_id]
    if not context_history:
        return  # Nothing to log
    prev_prompt, prev_response = context_history[-1]
    log_entry = {
        "prompt": prev_prompt,
        "bad_response": prev_response,
        "flagged_by": flag_phrase,
        "user_id": user_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + 'Z'
    }
    if not os.path.exists('logs'):
        os.makedirs('logs')
    with bad_log_lock:
        with open('logs/bad_responses.jsonl', 'a') as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
    logging.info(f"Bad response logged for review: {log_entry}")

def get_user_id() -> str:
    """Get a user/session ID from the request (for now, use IP as a simple stand-in)."""
    # For production, use a real session/token/cookie
    return request.remote_addr or 'default_user'


def get_faq_response(query: str) -> Optional[str]:
    """Return a FAQ/casual response using direct, regex, or semantic match."""
    normalized = query.strip().lower()
    # Direct match
    if normalized in faq_map:
        return faq_map[normalized]
    # Regex pattern match
    for pattern, answer in faq_patterns:
        if pattern.match(query.strip()):
            return answer
    # Embedding similarity match
    try:
        query_emb = embedder.encode([normalized], convert_to_numpy=True)
        scores = np.dot(faq_embeddings, query_emb.T).squeeze()
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > 0.8:  # Similarity threshold
            return faq_map[faq_questions[best_idx]]
    except Exception as e:
        logging.error(f"FAQ embedding similarity error: {e}")
    return None


def chatbot_response(query: str, user_id: str) -> str:
    """Generate a chatbot response, using context and robust intent detection. Also detect and log flagged bad responses."""
    # Check if the user is flagging a bad response
    for pattern in flag_patterns:
        if pattern.search(query):
            log_bad_response(user_id, query)
            # Optionally, you can return a special message here, but for now, just continue
            return "Thank you for your feedback. I'll try to do better next time!"
    # Use context window for this user
    context_history = user_contexts[user_id]
    faq_response = get_faq_response(query)
    if faq_response:
        response = faq_response
    else:
        try:
            # Build context for prompt
            context_lines = []
            for user_msg, bot_msg in context_history:
                context_lines.append(f"User: {user_msg}")
                context_lines.append(f"Assistant: {bot_msg}")
            # Add current user query
            context_lines.append(f"User: {query}")
            context = "\n".join(context_lines)
            # Retrieve relevant docs
            retrieved_docs = retrieve_documents(query, embedder, index)
            doc_context = "\n".join(doc[:300] for doc in retrieved_docs)
            prompt = f"{doc_context}\n{context}\nAssistant:"
            response = generate_response(prompt)
            # Post-process: Remove prompt echo, excessive length, and repeated content
            response = response.replace(prompt, '').strip()
            if response.lower().startswith(query.lower()):
                response = response[len(query):].strip()
            # Truncate to 2 sentences max
            response = re.split(r'(?<=[.!?]) +', response)
            response = ' '.join(response[:2]).strip()
            if not response:
                response = "I'm not sure how to answer that, but I'm here to help!"
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            response = "Sorry, I encountered an error. Please try again later."
    # Update context memory
    context_history.append((query, response))
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
    user_id = get_user_id()
    response = chatbot_response(query, user_id)
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
    # append_to_collected_data() # This function is no longer needed

# Remove append_to_collected_data definition and all its usages

def push_to_github():
    ensure_git_repo()  # Ensure the repo is cloned and set up

    # Get Git user name and email from environment variables
    git_email = os.environ.get('GIT_USER_EMAIL')
    git_name = os.environ.get('GIT_USER_NAME')
    token = os.environ.get('GITHUB_TOKEN')  # Ensure this is set in google cloud environment or .env

    if not git_email or not git_name or not token:
        print("GitHub credentials not set in environment variables. Please set GIT_USER_EMAIL, GIT_USER_NAME, and GITHUB_TOKEN.")
        return

    subprocess.run(['git', 'config', '--global', 'user.email', git_email])
    subprocess.run(['git', 'config', '--global', 'user.name', git_name])
    print("GitHub user configuration set.")

    # Configure Git to use the token for authentication
    subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'])
    with open(os.path.expanduser('~/.git-credentials'), 'w') as f:
        f.write(f'https://{token}:x-oauth-basic@github.com\n')
    print("GitHub token configured for authentication.")

    # Add all files to git
    subprocess.run(['git', 'add', '.'])
    print("Collected data added to Git staging area.")

    # Commit the changes
    subprocess.run(['git', 'commit', '-m', 'Collected data for finetuning'])
    print("Changes committed to Git.")

    # Push to the repository
    push_result = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True, text=True)  # Update 'main' if your branch is different
    if push_result.returncode == 0:
        print("Changes pushed to GitHub successfully.")
    else:
        print(f"Failed to push changes to GitHub: {push_result.stderr}")

# Schedule the job to run every hour (adjust as needed)
schedule.every(5).hours.do(push_to_github)

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
