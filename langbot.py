from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer


st.title("Language Bot")
st.write("This is a simple language bot that can answer questions based on the content of a PDF document.")

file = st.file_uploader("Upload a PDF file", type="pdf")
question = st.text_input("Ask a question")
k = 4

if file is not None:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    sentences = sent_tokenize(text)
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    sentence_embeddings = model.encode(sentences)
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, sentence_embeddings)
    top_k_similar = np.argsort(similarities[0])[-k:][::-1]
    most_similar = np.argmax(similarities)

    #st.write(f"Most similar sentence: {sentences[most_similar]}")
    #st.write(f"Top {k} similar sentences: {[sentences[idx] for idx in top_k_similar]}")

    top_k_similar_sentences = [sentences[idx] for idx in top_k_similar]
    input_text = f"Question: {question}\\nRelevant Sentences: {' '.join(top_k_similar_sentences)}\\nAnswer:"

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors='pt', padding=True)

    # Generate the answer
    outputs = model.generate(
        input_ids=inputs.input_ids,
        max_length=150,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )

    # Decode the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True) 
    st.write(f"Answer: {answer}")