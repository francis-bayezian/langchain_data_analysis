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
import pandas as pd
import numpy as np
import os
import streamlit as st


key = st.text_input("Enter your OpenAI key")
os.environ["OPENAI_API_KEY"] = key

user_input = st.text_input("Input your Query")

# File uploader for the parquet file
data_file = st.file_uploader("Upload your Data")

if data_file is not None:
    with open(os.path.join(os.getcwd(), data_file.name), 'wb') as f:
        f.write(data_file.getvalue())

# Check if a file has been uploaded
if data_file is not None:
    with st.spinner("Analyzing data...⌛"):
        data = pd.read_parquet(data_file)
        df_reset = data.reset_index()
        df_reset['symbol'] = df_reset['symbol'].ffill()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a AI financial stock market analyst. You are to provide a detail analysis of the data using the provided question"),
            ("user", "{input}")
            ])
        
        llm = create_pandas_dataframe_agent(ChatOpenAI(temperature=0,model = "gpt-4-0613"), df_reset, verbose=True)
        chain = prompt | llm
        chain.invoke({"input": f"calculate divergence by TSLA % change - SPY % change"})





