from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain.prompts import PromptTemplate, ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()
class SQL_LLM:
    def __init__(self):
        try:
            API_KEY=os.environ["API_KEY"]
            if 'history' not in st.session_state: #Used to pass into the LLM to remind previous conversations.
                st.session_state.history = []
            #instantiating the Gemini LLM
            self.llm = ChatGoogleGenerativeAI( 
                model="gemini-1.5-flash",
                temperature=0,
                google_api_key=API_KEY,
                
            )
            #Streamlit
            st.set_page_config(page_title="I can Retrieve Any SQL query")
            st.header("App To Retrieve SQL Data")
        except Exception as e:
            st.error("Error in initializing the app")
            return
    

    def get_sql_command(self, question, schema, history):
        """
        The function responsible for providing the SQL command based on the user's question. 
        Input: Schema of the database, history of the previous conversations."""
        try:
            #Template that pass to the LLM to provide answers as per the user request. 
            template = """
            As a PostgreSQL expert, generate a syntactically correct PostgreSQL query based on the input question, and return only the query without any extra text or backticks (```sql ```).

            Finally, Use only tables names and Column names mentioned in:\n\n {context} to create correct SQL Query and pay close attention on which column is in which table. if context contains more than one tables then create a query by performing JOIN operation only using the column unitid for the tables.\

            Question: {question}

            Do not use ``` , \n in begining or end of the SQL query.
            """
            prompt_template_one = ChatPromptTemplate.from_template(template)

        except Exception as e:
            st.error(f"Error in generating SQL command {e}")
            return ''




s=SQL_LLM()