from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("groq_api_key")

llm  = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.1-8b-instant")

# Initialize embeddings using the Hugging Face model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb_file_path = "faiss_index"


def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(
        file_path="codejay_chatbot_full_dataset.csv",
        source_column="Question",
        content_columns=["Answer"],
        metadata_columns=["Question"],
        encoding="utf-8",
        autodetect_encoding=True  # allow trying other encodings if UTF-8 fails
    )
    
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=embedder)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)
    
    

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embedder, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold = 0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only. In the answer try to provide as much text as possible from "response" section in the source document context without making much changes. If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer. No Preamble.
    
    CONTEXT: {context}
    
    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs={"prompt": PROMPT})

    return chain






