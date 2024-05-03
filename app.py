# load the dependencies
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Streamlit UI
st.title('Test App')
Userquestion = st.chat_input('Ask me a question')

if Userquestion:
    st.write(f'Your Question: {Userquestion}')

# define a function to create a llm model
from langchain_community.llms import HuggingFaceHub
def get_llm():
    hf = HuggingFaceHub(
        repo_id = 'google/gemma-1.1-7b-it',  # google/gemma-1.1-7b-it
        task = 'text-generation',
        model_kwargs = {'temperature':0.1, 'max_length':1000},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )
    return hf

# define a fucntion to load the pdf files (the data) from directory
from langchain_community.document_loaders import PyPDFDirectoryLoader
def load_data():
    pdf_loader = PyPDFDirectoryLoader('data')
    data = pdf_loader.load()
    return data

# define a function to split documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
def text_splitter(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    document_chunks = text_splitter.split_documents(document)
    return document_chunks

# define an embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings
def get_embeddingModel():
    model_name = 'BAAI/bge-small-en-v1.5'
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings':False}
    huggingFace_embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs,
    )
    return huggingFace_embeddings

# define a fucntion to embed the document chunks and store in a vector database
from langchain_community.vectorstores import FAISS
def get_vectorDB(document_chunks, embedding_model):
    vectorDB = FAISS.from_documents(document_chunks, embedding_model)
    return vectorDB

prompt_template = ''' 
Use the following piece of contect to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{question}

Helpful Asnwers:
'''
from langchain.prompts import PromptTemplate
def get_prompt(query):
    prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
    return prompt



from langchain.chains import RetrievalQA
def get_retrievalQA(vector_database,llm,prompt):
    retriever = vector_database.as_retriever(search_type='similarity', search_kwargs={'k':3})
    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={'prompt':prompt}
    )
    return retrievalQA



data = load_data()
document_chunks = text_splitter(data)
embedding_model = get_embeddingModel()
vectorDB = get_vectorDB(document_chunks, embedding_model)
llm = get_llm()



query = Userquestion
prompt = get_prompt(query)
retrievalQA = get_retrievalQA(vectorDB, llm, prompt)

if Userquestion:
    suggested_answer = retrievalQA.invoke({'query':query})
    st.write('Suggested answer:', suggested_answer['result'])