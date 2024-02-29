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
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()  # Load API keys and other sensitive data from the .env file 

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Invalid number of arguments: python main.py 'query'")
        sys.exit(1)  # Exit with error if incorrect arguments are given

    api_key = os.getenv("OPENAI_API_KEY")  # Retrieve the OpenAI API key from environment variables

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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create a retriever to search for relevant documents based on the user's query
    retriever = vector.as_retriever()
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how",
        "context":""
    })
    print(response["answer"])

 


