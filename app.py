import streamlit as st

import pandas as pd

from modules.conn import get_conn, get_engine

from Generation import Text_Generation_Revision, recommendation_engine

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from log_config import logging

personalities = pd.read_json("personalities.json")

persons = [i for i in personalities["person"]]
           
persons.append("Other")

st.markdown("<h1 style = 'text-align:center; color : red; font-size:39px;'> 🤖 AI Content Generator - Recommender</h1>", unsafe_allow_html = True)

st.sidebar.title("Content Creation")

LLM = st.sidebar.selectbox("Language Model", ["GPT", "Ollama", "Gemini"])

person = st.sidebar.selectbox("Personality Template", persons)

if person == "Other":

    person = st.text_input("Person")

    knowledge = st.text_input("Topic")

    submit = st.button("Generate Content")

else:

    knowledge = personalities[personalities["person"] == person].iloc[0, 2]

    submit = st.sidebar.button("Generate Content")

st.sidebar.title("Article Recommendation")
    
user_query = st.sidebar.text_input("Type a phrase, topic or content")

number = st.sidebar.number_input("Number of Articles to fetch", step = 1) # Take only int as input

search = st.sidebar.button("Search")

if search == True:
    logging.info("Search Similar Button Clicked")

    st.header(f"Top {number} Similar Articles")

    docs = recommendation_engine(user_query = user_query, k = number)

    for i in docs:
        st.text(i)


if submit == True:
    logging.info("Generate Button Clicked")
        
    content = Text_Generation_Revision(LLM, person, knowledge).generate_content()

    st.header(f"Author : {person} | Topic : {knowledge}")

    st.markdown(content)

    visualize = st.radio("Do you want to visualize the content ?", ['Yes', 'No'], index = 0)

    if visualize == "Yes":
        store = pd.read_sql_table("articles", con = get_engine())

        article_id = store[(store["person"] == person) & (store["topic"] == knowledge) & (store["model_used"] == LLM)].sort_values("created_at").iloc[-1, 0]

        metadata = pd.read_sql_table("metadata", con = get_engine())

        words = metadata[(metadata["publication_status"] == "Draft") & (metadata["article_id"] == article_id)].iloc[-1, 2]

        word_cloud = WordCloud(height = 200, width = 300, background_color = "white").generate(" ".join(words))

        figure, ax = plt.subplots(figsize = (10, 5))

        ax.imshow(word_cloud, interpolation = "bilinear")

        ax.axis("off")

        st.pyplot(figure)

    else:
        pass

