from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()
class SQL_LLM:
    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #embedding model
            self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
            self.retriever = self.vectorstore.as_retriever()
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
    

    def get_sql_command(self, question, history):
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
            retriever_prompt = ChatPromptTemplate.from_template(template)
            retriever_chain = (
                                {"context": self.retriever, "question": RunnablePassthrough()}
                                | retriever_prompt
                                | self.llm
                                | StrOutputParser()
                            )

        except Exception as e:
            st.error(f"Error in generating SQL command {e}")
            return ''
        
    def get_user_answer(self, question, sql_command, sql_answer):
        """
        This function will help to rewrite the answer provided by database to a user friendly manner. 
        Because after geting the SQL query from LLM teh database will give an answer and it will not be user firendly. """
        
        #template to rewrite the answer as per the user question.
        prompt_two = """
        You have given a user question to get details from a SQL database, sql command and the result from database. 
        You need to give a human friendly answer to the user. 
        The user question: {question},
        Output from database: {sql_answer}
        
        """
        prompt_template_two = ChatPromptTemplate.from_template(prompt_two)
        chain = prompt_template_two | self.llm #chain
        response = chain.invoke({'question':question, 'sql':sql_command, 'sql_answer':sql_answer}) #invoking of chain with the required parameters. 
        return response.content #extracting the answer only
    




s=SQL_LLM()
print(s.embeddings)