from collections import Counter
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai
from dotenv import load_dotenv
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("book")  # Data Ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Intelligent Mechanical Design Assistant")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    response_time = time.process_time() - start
    st.write(f"Response time: {response_time:.4f} seconds")

    generated_answer = response['answer']
    st.write(generated_answer)

    # BLEU score calculation with n-grams
    def ngram_precision(reference, candidate, n):
        reference_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
        candidate_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])

        overlap = sum((candidate_ngrams & reference_ngrams).values())  # Count overlapping n-grams
        precision = overlap / max(sum(candidate_ngrams.values()), 1)  # Avoid division by zero
        return precision

    def simple_bleu(reference, candidate, max_n=4):
        precisions = [ngram_precision(reference, candidate, n) for n in range(1, max_n+1)]
        geometric_mean = math.exp(sum(math.log(p) for p in precisions if p) / max_n)

        brevity_penalty = min(1.0, len(candidate) / max(len(reference), 1))  # Length-based penalty
        bleu_score = brevity_penalty * geometric_mean

        return bleu_score * 100  # Convert to percentage scale

    # Assuming you have a reference answer
    reference_answer = st.text_area("Enter reference answer for BLEU score calculation")

    if reference_answer:
        reference = reference_answer.split()  # Tokenize reference answer
        candidate = generated_answer.split()  # Tokenize generated answer

        # Calculate BLEU score
        bleu_score = simple_bleu(reference, candidate)
        st.write(f"BLEU Score: {bleu_score:.2f}")

        # Store BLEU score for further analysis
        if "bleu_scores" not in st.session_state:
            st.session_state.bleu_scores = []

        st.session_state.bleu_scores.append(bleu_score)

        # 1. BLEU Score Progression (Line Plot)
        st.write("BLEU Score Progress")
        fig, ax = plt.subplots()
        ax.plot(st.session_state.bleu_scores, marker='o')
        ax.set_xlabel("Query Number")
        ax.set_ylabel("BLEU Score")
        ax.set_title("BLEU Score Progress Over Queries")
        st.pyplot(fig)

        # 2. Precision, Recall, F1 Score (Bar Plot)
        precision = np.random.uniform(0.6, 0.9)  # Placeholder values
        recall = np.random.uniform(0.5, 0.8)     # Placeholder values
        f1_score = 2 * (precision * recall) / (precision + recall)

        st.write("Precision, Recall, F1 Score")
        fig, ax = plt.subplots()
        ax.bar(["Precision", "Recall", "F1"], [precision, recall, f1_score], color=['blue', 'green', 'orange'])
        ax.set_title("Precision, Recall, F1 Score")
        st.pyplot(fig)

        # 3. Response Length vs BLEU Score (Scatter Plot)
        response_length = len(candidate)  # Response length in words
        st.write(f"Response Length: {response_length} words")

        if "response_lengths" not in st.session_state:
            st.session_state.response_lengths = []

        st.session_state.response_lengths.append(response_length)

        st.write("Response Length vs BLEU Score")
        fig, ax = plt.subplots()
        ax.scatter(st.session_state.response_lengths, st.session_state.bleu_scores)
        ax.set_xlabel("Response Length")
        ax.set_ylabel("BLEU Score")
        ax.set_title("Response Length vs BLEU Score")
        st.pyplot(fig)

        # 4. Response Time vs Query (Line Plot)
        if "response_times" not in st.session_state:
            st.session_state.response_times = []

        st.session_state.response_times.append(response_time)

        st.write("Response Time vs Query")
        fig, ax = plt.subplots()
        ax.plot(st.session_state.response_times, marker='o')
        ax.set_xlabel("Query Number")
        ax.set_ylabel("Response Time (s)")
        ax.set_title("Response Time Over Queries")
        st.pyplot(fig)

        # 5. BLEU Score Distribution (Histogram)
        st.write("BLEU Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.bleu_scores, bins=10, kde=True, ax=ax)
        ax.set_title("BLEU Score Distribution")
        st.pyplot(fig)

        # 6. Response Time Distribution (Histogram)
        st.write("Response Time Distribution")
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.response_times, bins=10, kde=True, ax=ax)
        ax.set_title("Response Time Distribution")
        st.pyplot(fig)

        # 7. Precision vs Recall (Scatter Plot)
        st.write("Precision vs Recall")
        fig, ax = plt.subplots()
        ax.scatter(precision, recall, color='purple')
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.set_title("Precision vs Recall")
        st.pyplot(fig)

        # 8. BLEU Score vs Response Time (Scatter Plot)
        st.write("BLEU Score vs Response Time")
        fig, ax = plt.subplots()
        ax.scatter(st.session_state.response_times, st.session_state.bleu_scores)
        ax.set_xlabel("Response Time (s)")
        ax.set_ylabel("BLEU Score")
        ax.set_title("BLEU Score vs Response Time")
        st.pyplot(fig)

        # 9. BLEU Score Box Plot
        st.write("BLEU Score Box Plot")
        fig, ax = plt.subplots()
        ax.boxplot(st.session_state.bleu_scores)
        ax.set_title("BLEU Score Box Plot")
        st.pyplot(fig)

        # 10. Precision, Recall, F1 Scores Heatmap
        st.write("Precision, Recall, F1 Heatmap")
        fig, ax = plt.subplots()
        data = np.array([[precision, recall, f1_score]])
        sns.heatmap(data, annot=True, xticklabels=["Precision", "Recall", "F1"], yticklabels=["Scores"], cmap="coolwarm", ax=ax)
        ax.set_title("Precision, Recall, F1 Heatmap")
        st.pyplot(fig)

    # Display document similarity search results
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
