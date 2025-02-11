import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import pdfplumber
from docx import Document



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
    outputs = model.generate(inputs.input_ids, max_length=2000, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define chatbot response function
def chatbot_response(query):
    # Check for casual conversation
    casual_responses = {
        "how are you": "I'm just a program, but I'm here to help you! How can I assist you today?",
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! What can I do for you?"
    }
    
    # Normalize the query for easier matching
    normalized_query = query.lower().strip()
    
    # Check if the query matches a casual response
    if normalized_query in casual_responses:
        return casual_responses[normalized_query]
    
    # If not casual, proceed to retrieve documents
    retrieved_docs = retrieve_documents(query, embedder, index)
    prompt = construct_prompt(query, retrieved_docs)
    return generate_response(prompt)

# Set up Flask app
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")
    response = chatbot_response(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
