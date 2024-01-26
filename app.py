from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import streamlit as st

os.environ["OPENAI_API_KEY"] = "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"
def main():
    load_dotenv()


    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file")
    if csv_file is not None:
        with open(os.path.join(os.getcwd(), csv_file.name), 'wb') as f:
             f.write(csv_file.getvalue())
             
        data = pd.read_parquet(csv_file)
        document = data.reset_index()
        document['symbol'] = document['symbol'].ffill()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a AI financial stock market analyst. You are to provide a detail analysis of the data using the provided question"),
            ("user", "{input}")
            ])
        
        llm = create_pandas_dataframe_agent(ChatOpenAI(temperature=0,model = "gpt-4-0613"), document, verbose=False)
        chain = prompt | llm 

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(chain.invoke({"input": user_question}))
                
if __name__ == "__main__":
    main()




