# Intelligent-Mechanical-Design-Assistant
**Intelligent Mechanical Design Assistant (IMDA)** is an AI-powered tool for engineers to get precise answers from technical documents. It uses Groq, Llama3 models, FAISS embeddings, and Hugging Face for fast, document-based Q&amp;A. Ideal for quick design insights, it’s easy to use and planned for future CAD and simulation integrations.
![Chatbot Image](https://snapengage.com/wp-content/uploads/2020/11/bpos-survive-the-ai-revolution.jpg)


# Overview
The Intelligent Mechanical Design Assistant is a Streamlit application that leverages advanced language models to assist users in querying and analyzing research papers in mechanical engineering. The application utilizes Groq and Hugging Face embeddings to create a vector database, enabling efficient document retrieval and context-based question answering. Additionally, it evaluates the model's responses using the BLEU score and provides various visualizations to analyze performance metrics.

# Features
**Document Ingestion:** Load and process PDF documents from a specified directory.
**Contextual Querying:** Users can input questions, and the model provides answers based on the loaded documents.
**Performance Metrics:** The application calculates and visualizes BLEU scores, precision, recall, and F1 scores for the generated answers.
**Visual Analytics:** Several plots display trends and distributions of performance metrics over multiple queries.

# Technologies Used
**Streamlit:** For building the interactive web application.
**LangChain:** To facilitate the creation of language model chains and document loaders.
**FAISS:** For efficient similarity search and clustering of embeddings.
**Matplotlib and Seaborn:** For data visualization.
**OpenAI API:** For integrating with the language model.
**Python:** Programming language for developing the application.

# Installation
**1. Clone the Repository:**
```
git clone <your_repository_url>
cd <your_repository_name>
```

**2. Create a Virtual Environment (optional but recommended):**
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**3. Install Dependencies: Create a requirements.txt file with the following content:**
```
pip install -r requirements.txt
```

**4. Set Up Environment Variables:** Create a .env file in the root directory of your project and add your API keys in the following format:
```
GROQ_API_KEY=<your_groq_api_key>
HF_TOKEN=<your_hugging_face_token>
```
**Note:** Make sure to replace <your_groq_api_key> and <your_hugging_face_token> with your actual API keys.

**5. Run the Application: You can run the application using:**
```
streamlit run app.py  # Replace `app.py` with the name of your main Python file

```
