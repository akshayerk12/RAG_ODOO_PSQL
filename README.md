# AI-Powered SQL Query Chatbot for Odoo PostgreSQL Database
## Introduction:
This project involves developing an AI-powered SQL chatbot for a PSQL database based on user input. The chatbot enables users to ask complex, natural language questions about the database, and it generates the corresponding SQL queries. The project integrates Google’s Gemini-1.5-flash LLM, Hugginface Embeddings, Chroma Vector Database, and Streamlit to create a seamless user experience.

## Installation

Use the command to install the dependencies listed in a requirements.txt file: (Inside a Python  env)
```bash
pip install -r requirements.txt
```

## Usage

```python
streamlit run app.py
```
## But currently the project is not connected with the Odoo PSQL database, but can generate the SQL commands based on the table names and column names of the Odoo database with RAG. 
