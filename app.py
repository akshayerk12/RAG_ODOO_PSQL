from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import psycopg2

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
            self.password=os.environ["password"]
            # if 'history' not in st.session_state: #Used to pass into the LLM to remind previous conversations.
            #     st.session_state.history = []
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
    

    def get_sql_command(self, question):
        """
        The function responsible for providing the SQL command based on the user's question. 
        Input: Question by the user"""
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
            answer = retriever_chain.invoke(question)
            return answer

        except Exception as e:
            st.error(f"Error in generating SQL command {e}")
            return ''
        
    def get_user_answer(self, question, sql_answer):
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
        response = chain.invoke({'question':question, 'sql_answer':sql_answer}) #invoking of chain with the required parameters. 
        return response.content #extracting the answer only
    
    def read_sql_query(self, sql):
        """
        Function to connect to the PostgreSQL database and run the provided SQL command.
        Input: SQL command, database name, username, and password for the PostgreSQL database
        """
        try:
            # Connect to the PostgreSQL database
            conn = psycopg2.connect(
                dbname='odoo17', 
                user='odoo17', 
                password=self.password, 
                host='5.2.89.3',  
                port='5432'        
            )
            cur = conn.cursor()
            # Execute the provided SQL query
            cur.execute(sql)
            rows = cur.fetchall()  # Fetch all results of the query
            conn.commit()  # Commit the transaction (optional, depending on your needs)
        except psycopg2.Error as e:
            st.error(f"Database Error: {e}")
            rows = []
        except Exception as e:
            st.error(f"Error executing SQL: {e}")
            rows = []
        finally:
            if conn:
                conn.close()  # Close the database connection
            return rows



    def chat(self, prompt):
        """
        A single function to do all the tasks. All the functions are defined here in a step by step manner
        Input: The user question (directly from the chatbox)"""
        try:
            # st.session_state.history.append(prompt) #chat history
            sql_command = self.get_sql_command(prompt) #calling the function to get SQL command from LLM            
            st.write(sql_command) #To show the provided SQL command by the LLM (Not required for the user, but to make sure to the developer that the correct SQL query)           
            # sql_answer = self.read_sql_query(sql_command) #To get answer from Database

            # final = self.get_user_answer(prompt, sql_answer) #Change the answer as per the user question
            # print(st.session_state.history)
            # return final
        except Exception as e:
                st.error(f"Error occured {e}")
    




sql=SQL_LLM()
try:
    #streamlit app
    if prompt := st.chat_input("Ask a SQL question"): 
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from the chat method
        final = sql.chat(prompt)
        
        # Display LLM  response in chat message container
        with st.chat_message("ai"):
            st.markdown(final)
except Exception as e:
    st.error(f"Error in processing user input")
