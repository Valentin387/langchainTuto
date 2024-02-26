import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#access the API key from the .env file
from dotenv import load_dotenv
load_dotenv()
APIKEY = os.getenv("APIKEY")



if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Invalid number of arguments: python main.py 'query'")
        sys.exit(1)

    llm = ChatOpenAI(openai_api_key=APIKEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    print(chain.invoke({"input": "how can langsmith help with testing?"}))
    
    # Rest of your code goes here



