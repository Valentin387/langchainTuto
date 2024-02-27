import os
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}") 
    ])  # Define a prompt that instructs the AI to behave as a documentation writer

    output_parser = StrOutputParser()  # Create a parser to process plain text output
    chain = prompt | llm | output_parser  # Create the LangChain pipeline

    print(chain.invoke({"input": "how can langsmith help with testing?"})) 


