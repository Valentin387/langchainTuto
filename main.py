import os
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()  # Load API keys and other sensitive data from the .env file 

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Invalid number of arguments: python main.py 'query'")
        sys.exit(1)  # Exit with error if incorrect arguments are given

    api_key = os.getenv("APIKEY")  # Retrieve the OpenAI API key from environment variables

    # Ensure that the API key has been loaded before proceeding
    if not api_key:
        print("Error: API key not found in environment.")
        sys.exit(1)

    llm = ChatOpenAI(openai_api_key=api_key)  # Create the ChatGPT instance

    # Create a document loader to fetch documentation from the Langsmith website
    loader = WebBaseLoader("https://docs.smith.langchain.com")
    docs = loader.load()

    # Create an instance of the OpenAIEmbeddings class to generate embeddings for the documentation
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # Split the documentation into individual documents
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    <context>
    {context}
    </context>

    Question: {input}""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create a retriever to search for relevant documents based on the user's query
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"])

    # LangSmith offers several features that can help with testing:...

    
    """   
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}") 
    ])  # Define a prompt that instructs the AI to behave as a documentation writer

    output_parser = StrOutputParser()  # Create a parser to process plain text output
    chain = prompt | llm | output_parser  # Create the LangChain pipeline

    print(chain.invoke({"input": "how can langsmith help with testing?"}))  
    """


