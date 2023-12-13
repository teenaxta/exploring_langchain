from llm import get_few_shot_db_chain
import streamlit as st

if __name__=='__main__':

    st.title("🗄️ Query your Database")

    st.header("🪪 SQL Credentials")
    
    col1, col2 = st.columns(2)
    with col1:
        db_user=st.text_input("SQL_user", value="root")
        db_password=st.text_input("root", value="root")

    with col2:
        db_host=st.text_input('host', value="localhost")
        db_name=st.text_input("database name", value='atliq_tshirts')

    st.header("🤔 Question")
    question =st.text_input('Your question')

    if question:
        chain=get_few_shot_db_chain(db_user, db_password, db_host, db_name)
        answer=chain(question)
        st.header('Answer')
        st.write(answer['result'])
        st.header('SQL Query')
        st.write(answer['intermediate_steps'][1])