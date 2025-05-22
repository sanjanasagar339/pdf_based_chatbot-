from flask import Flask, request, jsonify
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import tempfile
import os
from flask_cors import CORS


pdf_path = "static/Course 2_ Leadership Habits and Characteristics.pdf"  # Move your PDF into a known location

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Reuse your original functions here: load_pdf_chunks, get_embedding, etc.

def load_pdf_chunks(file_path, chunk_size=500):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    sentences = full_text.split('. ')
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
        return chunks 
# ... include your other functions like get_embedding, etc. ...
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response["data"][0]["embedding"]

def get_pdf_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embeddings.append(get_embedding(chunk))
    return np.array(embeddings)

chunks = load_pdf_chunks(pdf_path)
embeddings = get_pdf_embeddings(chunks)

def search_chunks(query, chunks, embeddings, top_k=3, threshold=0.75):
    query_embedding = get_embedding(query)
    sims = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    if max(sims[top_indices]) < threshold:
        return None
    return [chunks[i] for i in top_indices]

def ask_chatgpt(context, query):
    prompt = f"""
You are an assistant that answers questions strictly based on the following PDF content:

{context}

Only use the above material to answer the question. If the question is unrelated, reply:
"I'm only able to answer questions related to the uploaded document. Please ask something relevant."

Question: {query}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
# Flask route to handle the chat request 
@app.route('/chat', methods=['POST'])
def chat():
    try:
        query = request.form.get('query')

        if not query:
            return jsonify({"error": "Missing 'query' in the request"}), 400

        # Greet if user says hi/hello/etc.
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        if query.strip().lower() in greetings:
            return jsonify({"answer": "Hello! How can I assist you with the document today?"})

        # Continue with PDF-based logic
        results = search_chunks(query, chunks, embeddings)

        if results:
            context = "\n\n".join(results)
            answer = ask_chatgpt(context, query)
        else:
            answer = "I'm only able to answer questions related to the uploaded document. Please ask something relevant."

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)