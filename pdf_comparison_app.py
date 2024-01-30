import streamlit as st
import os
import tempfile
from pathlib import Path
from pydantic.v1 import BaseModel, Field
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent
import openai

# Function to load variables from .env file
def load_env(file_path=".env"):
    with open(file_path) as f:
        for line in f:
            # Ignore comments and empty lines
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# Load environment variables from .env
load_env()
openai_api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Check if the API key is set
if not openai_api_key:
    raise ValueError("OpenAI API key is not set. Please check your .env file.")
if not pinecone_api_key:
    raise ValueError("Pinecone API key is not set. Please check your .env file.")

# Now you can use the retrieved API key in your code
openai.api_key = openai_api_key
OpenAIEmbeddings.api_key = openai_api_key
#pc = Pinecone(api_key=pinecone_api_key)
    
class DocumentInput(BaseModel):
    question: str = Field()

# Create a temporary directory in the script's folder
script_dir = Path(__file__).resolve().parent
temp_dir = os.path.join(script_dir, "tempDir")

def main():
    st.title("PDF Document Comparison")

    # Create a form to upload PDF files and enter a question
    st.write("Upload the first PDF file:")
    pdf1 = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf1")

    st.write("Upload the second PDF file:")
    pdf2 = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf2")

    question = st.text_input("Enter your question")
    submit_button = st.button("Compare PDFs")

    if submit_button:
        if pdf1 and pdf2:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            else:
                # Clear the previous contents of the "tempDir" folder
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error deleting file: {e}")

            # Save the PDF files to the "tempDir" directory
            pdf1_path = os.path.join(temp_dir, pdf1.name)
            with open(pdf1_path, 'wb') as f:
                f.write(pdf1.getbuffer())

            pdf2_path = os.path.join(temp_dir, pdf2.name)
            with open(pdf2_path, 'wb') as f:
                f.write(pdf2.getbuffer())



            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

            tools = []
            files = [

                {
                    "name": pdf1.name,
                    "path": pdf1_path,
                },

                {
                    "name": pdf2.name,
                    "path": pdf2_path,
                },
            ]

            for file in files:
                loader = PyPDFLoader(file["path"])
                pages = loader.load_and_split()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                docs = text_splitter.split_documents(pages)
                embeddings = OpenAIEmbeddings()
                retriever = FAISS.from_documents(docs, embeddings).as_retriever()

                # Wrap retrievers in a Tool
                tools.append(
                    Tool(
                        args_schema=DocumentInput,
                        name=file["name"],
                        description=f"useful when you want to answer questions about {file['name']}",
                        func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
                    )
                )
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                verbose=True,
                handle_parsing_errors=True
            )

            st.write(agent({"input": question}))
            # Now you have both PDFs saved in the "tempDir" folder
            # You can perform your PDF comparison here


if __name__ == "__main__":
    main()