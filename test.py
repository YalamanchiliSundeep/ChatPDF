import os
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pickle
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Set the Streamlit page configuration at the very top
st.set_page_config(page_title="Chat with PDF & DOCX Files 💁")

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API Key not found. Please set the API key in the environment.")

# Cache folder and file listings
@st.cache_data
def list_files_in_upload(files):
    """List all PDF and DOCX files uploaded by the user."""
    return [file for file in files if file.name.endswith('.pdf') or file.name.endswith('.docx')]

# Extract text from DOCX files
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    full_text = [paragraph.text for paragraph in doc.paragraphs]
    text_with_page_nums = [(text, docx_file.name, idx) for idx, text in enumerate(full_text) if text.strip()]
    return text_with_page_nums

# Extract text from PDF files
def extract_text_from_pdf(file):
    text = []
    try:
        pdf_reader = PdfReader(file)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append((page_text, file.name, page_num))
        return text
    except Exception as e:
        st.error(f"Error reading PDF {file.name}: {e}")
    return text

# Extract text from either PDF or DOCX files
def extract_text(file):
    if file.name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file.name.endswith('.docx'):
        return extract_text_from_docx(file)
    else:
        st.error(f"Unsupported file type for {file.name}. Only PDF and DOCX files are supported.")
        return []

# Split text into manageable chunks
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for page_text, file_name, page_num in text:
        splits = text_splitter.split_text(page_text)
        chunks.extend([(split, file_name, page_num) for split in splits])
    return chunks

# Create a vector store for text chunks using OpenAI embeddings
def get_vector_store(text_chunks, index_path="faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = [chunk[0] for chunk in text_chunks]
    
    if os.path.exists(f"{index_path}.index"):
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts(texts)
    else:
        vector_store = FAISS.from_texts(texts, embeddings)
    
    vector_store.save_local(index_path)
    
    with open(f"{index_path}/page_numbers.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

# Conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    You are an expert assistant. Use the provided context to answer the question as accurately as possible.
    If the information is not available in the context, state that the answer is not available in the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatOpenAI(api_key=openai_api_key, temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user input and generate a response
def process_user_input(user_question):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Load FAISS index with dangerous deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question, k=5)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    with open("faiss_index/page_numbers.pkl", "rb") as f:
        text_chunks = pickle.load(f)

    response_with_pages = []
    for doc in docs:
        for chunk, file_name, page_number in text_chunks:
            if doc.page_content in chunk:
                response_with_pages.append((file_name, page_number))
                break

    st.write("Reply:", response["output_text"])
    unique_pages = sorted(set(response_with_pages))
    st.write("Source(s):")
    for file_name, page_number in unique_pages:
        st.write(f"- {file_name}, Page: {page_number + 1}")

# Process the uploaded files, extract and vectorize content
def process_files(files, chunk_size=1000, chunk_overlap=200):
    all_text_chunks = []
    
    def process_file(file):
        st.write(f"Processing file: {file.name}")
        text_data = extract_text(file)
        if text_data:
            text_chunks = get_text_chunks(text_data, chunk_size, chunk_overlap)
            return text_chunks
        return []

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, files)
        for result in results:
            all_text_chunks.extend(result)
    
    if all_text_chunks:
        get_vector_store(all_text_chunks)
        st.success("Files processed and vector store created.")
    else:
        st.error("No text data found in the uploaded files.")

def main():
    st.header("Chat with PDF & DOCX Files 💁")

    # Input field for user's question
    user_question = st.text_input("Ask a Question")

    if user_question:
        process_user_input(user_question)

    with st.sidebar:
        st.title("Menu:")

        chunk_size = st.sidebar.number_input("Chunk Size", min_value=200, max_value=2000, value=1000)
        chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=50, max_value=500, value=200)

        # File uploader to allow users to upload multiple PDF/DOCX files
        uploaded_files = st.sidebar.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

        if uploaded_files:
            selected_files = list_files_in_upload(uploaded_files)

            st.write("Uploaded Files:")
            for file in selected_files:
                st.write(f"- {file.name}")

            # Only process the files when this button is clicked
            if st.sidebar.button("Process Uploaded Files"):
                if selected_files:
                    with st.spinner("Processing files..."):
                        process_files(selected_files, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                else:
                    st.error("No files selected for processing.")

if __name__ == "__main__":
    main()
    