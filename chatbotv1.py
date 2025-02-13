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
    return f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer as Arsene:"

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
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        return casual_responses["greeting"]
    # Use regex to match status inquiry
    elif re.search(r'\b(how are you|what\'s up|status)\b', normalized_query):
        return casual_responses["status"]
    # Use regex to match farewell
    elif re.search(r'\b(goodbye|bye|farewell)\b', normalized_query):
        return casual_responses["farewell"]
    # Use regex to match thanks
    elif re.search(r'\b(thank you|thanks)\b', normalized_query):
        return casual_responses["thanks"]
    # Use regex to match how are you
    elif re.search(r'\b(how are you|how do you do)\b', normalized_query):
        return casual_responses["how are you"]
    retrieved_docs = retrieve_documents(query, embedder, index)
    prompt = construct_prompt(query, retrieved_docs)
    return generate_response(prompt)

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

# Error handler for 502 errors
@app.errorhandler(502)
def bad_gateway_error(error):
    return jsonify({"error": "Bad Gateway. Please try again later."}), 502

if __name__ == "__main__":
    import os
    # For production using Gunicorn, the PORT will be managed via Procfile.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
