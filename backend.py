import sys
import os
import requests
import groq
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_PROVIDER = "gemini" # Options: "local", "groq", "gemini"
FAISS_PATH = "vector_db"


def get_rag_retriever():
    """Loads the pre-existing FAISS vector database from disk."""

    if not os.path.exists(FAISS_PATH):
        print(f"Vector database not found at {FAISS_PATH}. Please run create_vector_db.py first.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    
    # Load the vector store from the local folder
    vector_store = FAISS.load_local(
        FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True # Required for loading local FAISS indexes
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 3})

def get_llm_response(prompt_content):

    if LLM_PROVIDER == "local":
        return call_local_ollama(prompt_content)
    elif LLM_PROVIDER == "groq":
        return call_groq_api(prompt_content)
    elif LLM_PROVIDER == "gemini":
        return call_gemini_api(prompt_content)
    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}")

def call_local_ollama(prompt_content):
    payload = {"model": "llama3.2", "messages": [{"role": "user", "content": prompt_content}], "stream": False, "options": {"temperature": 0.0}}
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()
        return response.json()['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error contacting local Ollama model: {e}"

def call_groq_api(prompt_content):
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt_content}], model="llama3-8b-8192", temperature=0.0)
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error contacting Groq API: {e}"

def call_gemini_api(prompt_content):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)
        response = llm.invoke(prompt_content)
        return response.content
    except Exception as e:
        return f"Error contacting Gemini API: {e}"

def main():
    user_prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    # user_prompt = "what is my mac address and how do I change it?"

    retriever = get_rag_retriever()
    
    retrieved_context = ""

    if retriever:
        retrieved_docs = retriever.invoke(user_prompt)
        retrieved_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = f"""
        You are a Windows Command Line Assistant. Your only function is to create a single, executable command to fulfill a user's request using only modern, built-in Windows tools like CMD.exe or PowerShell.

        ## Rules:
        1.  **Use Modern Tools:** Prioritize modern and non-deprecated commands. For Python packages, ALWAYS prefer using `python -m pip install <package_name>`. 
        2.  **Create Robust Commands:** For software installations, your command MUST be a single-line PowerShell command that uses a `try...catch` block.
            * The `try` block should attempt the simplest installation method (e.g., `python -m pip install Django`).
            * The `catch` block should contain the fallback method: use `Invoke-WebRequest` to download the source, use `tar -xzf` to extract it, and then use `python -m pip install ./<extracted_folder>` to install it locally.
        3.  **Raw Output:** Your response must be the raw command text ONLY. Do not include markdown or explanations.
        4.  **Failure Handling:** If a task is truly impossible, output an `echo` command that concisely explains the limitation.

        ---
        ## Context & Task

        Use the provided `[Informational Context]` to help generate the command for the `[User Request]`.

        [Informational Context]:
        {retrieved_context}

        [User Request]:
        "{user_prompt}"
        """
    generated_command = get_llm_response(final_prompt)

    print(user_prompt, "\n\n", "--------------------------------\n", generated_command.strip())
    

if __name__ == "__main__":
    main()
