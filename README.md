# Arsène Manzi's Virtual Assistant Chatbot

## Overview

I built this virtual assistant chatbot to provide quick, conversational responses about my background, skills, and projects. The chatbot is designed to retrieve context from a custom-curated dataset (stored in `curated_data.json`) using a Retrieval-Augmented Generation (RAG) approach. It uses a fine-tuned GPT-Neo model to generate personalized, first-person responses that reflect my academic, professional, and personal experiences.

## Features

- **Conversational AI:** Engages in dialogue and answers questions in a natural, friendly manner.
- **Custom Curated Dataset:** I maintain a rich dataset with diverse categories (Biography, Education, Work Experience, Projects, Skills, Philosophy, Certifications, Interests, Achievements, Goals, Hobbies, Languages, Extracurricular Activities, and Values) stored in `curated_data.json`. This dataset gives the model context about my background.
- **Retrieval-Augmented Generation (RAG):** When a query is made, the system retrieves the most relevant snippets from my dataset and uses them to generate a well-informed response.
- **Error Handling and Logging:** I log the prompts and responses for continuous improvement and troubleshooting.
- **Deployment Ready:** The project is set up with Flask and Gunicorn for deployment on platforms like Railway, with proper CORS handling for integration with my portfolio site.


## Requirements

- Python 3.7 or higher
- `sentence-transformers`
- `faiss-cpu`
- `python-docx`
- `Flask`
- `Flask-Cors`
- `PyPDF2`
- `pdfplumber`
- `numpy`
- `requests`
- `scikit-learn`
- `torch`
- `transformers`
- `gunicorn`

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Setup and Deployment

### Clone the Repository:
```bash
git clone <repository-url>
cd project-directory
```

### Curate the Data:

I maintain my personal and professional background information in curated_data.json. This file is structured in a modular, first-person format and includes multiple categories such as Biography, Education, Work Experience, Projects, Skills, and more. The RAG system uses this file to retrieve context when generating responses.

### Run Locally:

To test the chatbot locally:

```bash
python chatbotv1.py
```
### Deploying on Railway:

My project is configured to run on Railway using Gunicorn. I use a Procfile with the following content:

```bash
web: gunicorn chatbotv1:app --bind 0.0.0.0:$PORT
```

When I push changes to my Git repository, Railway automatically redeploys my app. Ensure that the environment variable PORT is correctly set by Railway (usually 8080).

### CORS and Integration:

    I enable CORS in my Flask app so that my chatbot API can be accessed from my portfolio site without cross-origin issues. My portfolio widget uses JavaScript to interact with the deployed API.

Usage

After deploying, my chatbot API is available at a public URL (e.g., https://me0-minimvp-production.up.railway.app/chat). I integrate this endpoint into my portfolio as a chat widget. When a user submits a query, the following happens:

#### Input Processing:

    The query is normalized and, if it’s a casual greeting (e.g., "hi", "hello"), a pre-defined response is returned immediately.

#### Context Retrieval:

    If the query is not a simple greeting, my system computes embeddings for the query and retrieves the most relevant snippets from curated_data.json using FAISS.

#### Response Generation:

    The retrieved snippets are combined with the query to form a prompt, which is then fed to the GPT-Neo model to generate a personalized response.

#### Logging:

    The prompt and the generated response are logged for future refinement and analysis.

## Customization and Fine-Tuning

#### Dataset Updates:

    I continuously update curated_data.json as I gain new experiences, skills, and insights.

#### Model Fine-Tuning:

    For further improvements, I have an optional refinement/ directory with scripts and a dataset (cleaned_data.json) for fine-tuning the model on my specific conversational style and desired responses(more info can be found in the README.md file inside the refinement dir)

#### Parameter Tuning:
    You can adjust parameters such as max_new_tokens in the response generation function to balance between response length and latency.

### Contributing

This project reflects my personal journey and experiences. If you have suggestions or improvements, feel free to open an issue or submit a pull request. I welcome contributions that help refine and extend the capabilities of my virtual assistant.
License

This project is licensed under the MIT License.




