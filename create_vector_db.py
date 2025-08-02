import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


DATA_DIR = "data/"
FAISS_PATH = "vector_db"


def create_vector_db():

    #Use UnstructuredYAML Loader for .yml files
    loader = DirectoryLoader(DATA_DIR, glob="**/*.yml", loader_cls=UnstructuredFileLoader)
    
    #Load all documents from the specified directory using the loader
    documents = loader.load()
    
    #Initialize a text splitter to break documents into chunks of 1000 characters with 100 character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    #Split the loaded documents into smaller text chunks
    texts = text_splitter.split_documents(documents)

    #Create an embeddings object using the Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    
    #Build a FAISS vector store from the text chunks and their embeddings
    vector_store = FAISS.from_documents(texts, embeddings)

    #Save the created FAISS vector store to the specified local path
    vector_store.save_local(FAISS_PATH)

    print("Vector database created successfully.")

if __name__ == "__main__":
    create_vector_db()


