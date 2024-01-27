from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PIL import Image
import pandas as pd
import numpy as np
import os
import streamlit as st




def main():
    st.set_page_config(page_title="Ask your CSV")
    st.title("Financial Analyzer 📊")
    key = st.text_input("Enter your OpenAI key",type="password")
    os.environ["OPENAI_API_KEY"] = key

    csv_file = st.file_uploader("Upload a CSV file")
    if csv_file is not None:
        with open(os.path.join(os.getcwd(), csv_file.name), 'wb') as f:
             f.write(csv_file.getvalue())
             
        data = pd.read_parquet(csv_file)
        document = data.reset_index()
        document['symbol'] = document['symbol'].ffill()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a AI financial stock market analyst. You are to provide a detail analysis of the data using the provided question."),
            ("user", "{input}")
            ])
        
        llm = create_pandas_dataframe_agent(ChatOpenAI(temperature=0,model = "gpt-4-0613"), document, verbose=False)
        chain = prompt | llm
        
        
        user_question = st.text_input("Ask a question about your CSV: ")
        question =  user_question + f" . Plot and save the outputs"

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                output = chain.invoke({"input": question })
                desired_output = output['output']
                message = st.chat_message("assistant")
                message.write(desired_output)
                
                current_directory = os.getcwd()
                files = os.listdir(current_directory)
                image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                for image_file in image_files:
                    image_path = os.path.join(current_directory, image_file)
                    image = Image.open(image_path)
                    st.image(image, caption=image_file, use_column_width=True)
                    
                csv_files = [f for f in os.listdir(current_directory) if f.endswith('.csv')]
                for csv_file in csv_files:
                    csv_path = os.path.join(current_directory, csv_file)
                    data_csv = pd.read_csv(csv_path)
                    st.table(data_csv)
        
                    
                files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif','.csv','.pq'))]
                for file in files:
                    image_path = os.path.join(current_directory, file)
                    os.remove(image_path)
            
if __name__ == "__main__":
    main()




