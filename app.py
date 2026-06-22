import os
import gradio as gr

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Global variables to store our RAG components once initialized
rag_chain = None

def initialize_rag(api_key):
    """Parses the PDF document and sets up the RAG pipeline."""
    global rag_chain
    
    if not api_key.strip():
        # Hardcoded fallback key (Be careful exposing API keys in production code)
        api_key = "AQ.Ab8RN6JhzMnoz3buDzJDVp2BzpC5dVIhXtph4TWwXJdrb1dfyw"
    
    pdf_filename = "NVIDIAAn.pdf"  # Ensure this file is in your workspace folder
    if not os.path.exists(pdf_filename):
        return f"❌ Error: '{pdf_filename}' not found in the current directory."

    try:
        # Set environment variable for Gemini
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # 1. Load and split PDF
        loader = PyPDFLoader(pdf_filename)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 2. Vector Embeddings & Local Chroma Database
        embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # 3. Setup LLM and System Prompts (FIXED: Using gemini-2.5-flash)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        system_prompt = (
            "You are an expert financial analyst assistant. Use the following pieces of retrieved context "
            "from NVIDIA's Q1 Fiscal 2026 earnings release to answer the user's question.\n"
            "If you don't know the answer, state that you cannot find it in the document. Do not make things up.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 4. Construct RAG Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return "✅ System successfully initialized! You can now chat with the document below."
    except Exception as e:
        return f"❌ Initialization failed: {str(e)}"

def predict(message, history):
    """Handles conversational interactions from gr.ChatInterface."""
    global rag_chain
    if rag_chain is None:
        return "Please input your Gemini API Key and initialize the system first using the panel above."
    
    try:
        response = rag_chain.invoke({"input": message})
        return response["answer"]
    except Exception as e:
        return f"Error gathering data: {str(e)}"

# --- Gradio UI Layout Building ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 NVIDIA Q1 Fiscal 2026 Financial Results RAG")
    gr.Markdown("Query information regarding revenue performance, export constraints, and operational guidelines.")
    
    # Configuration Box
    with gr.Row():
        with gr.Column(scale=4):
            api_key_input = gr.Textbox(
                label="Gemini API Key", 
                placeholder="AIzaSy...", 
                type="password"
            )
        with gr.Column(scale=1):
            init_btn = gr.Button("Initialize System", variant="primary")
            
    status_output = gr.Markdown("⚠️ Status: System uninitialized. Enter API key to start.")
    
    # Event link to trigger setup
    init_btn.click(fn=initialize_rag, inputs=[api_key_input], outputs=[status_output])
    
    gr.Markdown("---")
    
    # Native Chat Interface
    gr.ChatInterface(
        fn=predict,
        examples=[
            "What was the total revenue for Q1 Fiscal 2026?", 
            "What was the financial impact of the H20 export control restriction?",
            "What are the expectations for Q2 FY2026 gross margins?"
        ]
    )

# Run the system with a shareable public link
if __name__ == "__main__":
    demo.launch(share=True)