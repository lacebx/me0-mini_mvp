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

# Pre-warm the model
model.eval()  # Set the model to evaluation mode
dummy_input = tokenizer("dummy input", return_tensors="pt")  # Create a dummy input
_ = model.generate(dummy_input.input_ids)  # Run dummy inference

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
    outputs = model.generate(inputs.input_ids, max_length=2000, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)  # Changed num_beams to num_return_sequences
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define chatbot response function
def chatbot_response(query):
    # Check for casual conversation
    casual_responses = {
        "how are you": "I'm just a program, but I'm here to help you! How can I assist you today?",
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! What can I do for you?",
        "what can you do": "I can assist you with a wide range of tasks, from answering questions to generating text. How can I help you today?",
        "what's your purpose": "My purpose is to assist users like you with information and tasks. I'm here to make your life easier!",
        "can you help me": "Of course! I'll do my best to assist you with whatever you need. Please let me know how I can help.",
        "how do you work": "I'm a complex system that uses AI and machine learning to understand and respond to user queries. I'm constantly learning and improving!",
        "are you a human": "No, I'm not a human. I'm a computer program designed to simulate conversation and answer questions to the best of my ability.",
        "can you understand sarcasm": "I'm getting better at understanding sarcasm, but I'm not perfect yet. Please bear with me if I don't always catch the tone!",
        "can you tell jokes": "I can try! I have a vast collection of jokes. Would you like to hear one?",
        "can you summarize": "Yes, I can summarize long pieces of text into shorter, more digestible versions. Please provide the text you'd like me to summarize.",
        "can you translate": "I can translate text from one language to another. Please provide the text and the languages you'd like to translate between.",
        "can you write a story": "I'd be happy to generate a story for you. Do you have any specifications or prompts in mind?",
        "can you chat": "I'm happy to chat with you about any topic you'd like. I'm a good listener and can respond thoughtfully.",
        "can you play games": "I can play simple text-based games with you. Would you like to play a game of Hangman or 20 Questions?",
        "can you solve math problems": "I can solve a wide range of math problems, from simple algebra to advanced calculus. Please provide the problem you'd like me to solve.",
        "can you convert units": "I can convert units of measurement, time, and currency. Please provide the value and the units you'd like to convert.",
        "can you give advice": "I can provide general advice on a wide range of topics. Please keep in mind that my advice is not a substitute for professional advice.",
        "can you recommend": "I can recommend books, movies, music, and more based on your preferences. Please let me know what you're interested in, and I'll do my best to suggest something.",
        "can you explain": "I can explain complex topics in simple terms. Please let me know what you'd like me to explain, and I'll do my best to break it down for you."
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

@app.route("/", methods=["GET"])
def home():
    return "Hello, world! The I'm running."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")
    response = chatbot_response(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
