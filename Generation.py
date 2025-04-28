from langchain_ollama import ChatOllama

from langchain.prompts import ChatPromptTemplate, PromptTemplate

from langchain_openai import ChatOpenAI

from langchain.schema import HumanMessage

from modules.conn import get_conn, run_query, get_engine

from nltk.corpus import stopwords

import string

import re

import textstat

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

from datetime import datetime

from sentence_transformers import SentenceTransformer

import pickle

from chromadb import PersistentClient

from chromadb.config import Settings

import sqlite3

import json

from log_config import logging

from nltk.tokenize import sent_tokenize

import pandas as pd

import os

load_dotenv()

os.environ["CHROMA_TELEMETRY"] = "False" # To not record the chroma db anonymous events in log file

api = os.getenv("gemini_api")

stp = stopwords.words("english")

stp.extend([i for i in string.punctuation])

lemma = WordNetLemmatizer()

content = []

client = PersistentClient(path = "chroma_db") 

collection = client.get_or_create_collection(name = "generated_content_embeddings")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class Text_Generation_Revision:
    '''Generates the content and can revise the generated content'''

    def __init__(self, model_name : str, person : str, topic : str):

        self.model_name = model_name
        self.person = person
        self.topic = topic

    def generate_content(self):
        '''To generate the content based on user input'''

        logging.info(f"Generating Content for {self.person}")

        store = pd.read_sql_table("articles", con=get_engine())

        data = pd.read_json("personalities.json")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system","You are {person} having knowledge in {topic}."),
            ("human", "Write an medium format article attaching relevant links at the end in {style}.")])

        inf = store[(store["topic"] == self.topic) & (store["person"] == self.person) & (store["model_used"] == self.model_name)]["content"]

        if inf.shape[0] != 0:

            if inf.shape[0] != 1:

                #print("".join(inf.sort_values("created_at").iloc[-1]))

                content_ = "".join(inf.sort_values("created_at").iloc[-1])

            else:
                #print("".join(inf.iloc[-1]))

                content_ = "".join(inf.iloc[-1])

            logging.debug(f"Content has been successfully generated for {self.person} using {self.model_name}")

        else:
            if self.model_name == "GPT":

                model = ChatOpenAI(model = "gpt-4o")

            elif self.model_name == "Ollama":

                model = ChatOllama(model = "phi")

            elif self.model_name == "Gemini":

                model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro", api_key = api)

            chain = prompt_template | model | StrOutputParser()
            try:

                df = data[data["person"] == self.person]

                number = int(df["Serial No"][df.index[0]])

                style = df["style"][df.index[0]]

                for chunk in chain.stream({"person":self.person, "topic":self.topic, "style":style}):

                    #print(chunk, end = "")

                    content.append(chunk)

                    now = datetime.now()

                logging.debug(f"Content has been successfully generated for {self.person}")

            except IndexError:
                logging.error(f"{self.person} not found in existing personalities")

                style = "in a clear and practical way"

                number = 0

                for chunk in chain.stream({"person":self.person, "topic":self.topic, "style":"in a clear and practical way"}):

                    #print(chunk, end = "")

                    content.append(chunk)

                    now = datetime.now()

                logging.debug(f"Content has been successfully generated for {self.person}")
                    
            content_ = "".join(content)

            values = []

            if number:
                values.append(number)

            elif 11 not in list(store["article_id"]):
                number = 11

                values.append(number)

            else:
                number = int(store["article_id"].max()) + 1

                values.append(number)

            sentences = sent_tokenize(re.sub("http[s]\:\/\/.+? +", "", content_))

            sentence_embeddings = embed_model.encode(sentences)

            all_items = collection.get()["ids"]

            id = []

            for i in all_items[-1]:

                if i.isdigit():

                    id.append(i)


            id_num = int("".join(id)) + 1

            id_list =  [f"Id{i}" for i in range(id_num, id_num + len(sentences))]

            collection.add(ids = id_list,
                           documents = sentences, embeddings = [sentence_embeddings[i] for i in range(len(sentences))])
            
            logging.info("Data has been successfully stored in Chroma DB")
            
            chroma_ids = []

            for i in id_list:

                dup_ids = []

                for j in i:

                    if j.isdigit():

                        dup_ids.append(j)

                chroma_ids.append(int("".join(dup_ids)))

            for i,j in zip(id_list, chroma_ids):

                sent_id = j

                chroma_sent = collection.get(ids = i)["documents"][0]

                chroma_embed = collection.get(ids = i, include = ["embeddings"])["embeddings"]

                query = "insert into sentence_embeddings values (%s, %s, %s);"

                values_ = (sent_id, chroma_sent, chroma_embed)

                run_query(query = query, values = values_)

            logging.debug("Data has been stored into sentence embeddings")

            binary_embeddings = pickle.dumps(sentence_embeddings)

            values.extend(["Understanding" + " " + self.topic,
                        json.dumps(sentences),
                        self.topic,
                        self.person,
                        style,
                        self.model_name,
                        now.strftime("%Y-%m-%d %H:%M:%S"),
                        binary_embeddings])

            query = "insert into articles values (%s, %s, %s, %s, %s, %s, %s, %s, %s);"

            run_query(query = query, values = values)

            logging.debug("Data has been successfully stored in articles table")

            meta_data = pd.read_sql_table("metadata", con = get_engine())

            values = []

            if meta_data["metadata_id"].shape[0] == 0:

                values.append(1)

            else:

                values.append(meta_data["metadata_id"].max() + 1)

            values.append(number)

            metacontent = [lemma.lemmatize(word) for word in word_tokenize(content_) if word.lower() not in stp and not word.isdigit()]

            reading_time = round(len(word_tokenize(content_))/200)

            publication_status = "Draft"

            values.extend([json.dumps(metacontent), reading_time, publication_status])

            query = "insert into metadata values (%s, %s, %s, %s, %s)"

            values = tuple(values)

            run_query(query = query, values = values)

            logging.debug("Data has been stored in metadata table")

            model = pd.read_sql_table("model_performance", con = get_engine())

            values = []

            if model["Test_id"].shape[0] == 0:

                values.append(1)

            else:
                values.append(model["Test_id"].max() + 1)

            values.extend([number, self.model_name])

            readability = textstat.flesch_reading_ease(content_)

            score = round(readability / 100, 1)

            now = datetime.now()

            values.extend([now.strftime("%Y-%m-%d %H:%M:%S"), score, publication_status])

            query = "insert into model_performance values (%s, %s, %s, %s, %s, %s);"

            run_query(query = query, values = tuple(values))

            logging.debug("Data has been successfully stored in model performance table")

            personality = pd.read_sql_table("personality_templates", con = get_engine())

            values = []

            if personality["template_id"].shape[0] == 0:

                values.append(1)

            else:

                values.append(personality["template_id"].max() + 1)

            values.extend([number, self.person])

            if style.startswith("in"):

                values.append("Explaining " + self.topic + " " + style)

                values.append(f"System : You are {self.person} having knowledge in {self.topic}\nHuman : Write an article {style}")

            else:

                values.append("Explaining " + self.topic + " in " + style)

                values.append(f"System : You are {self.person} having knowledge in {self.topic}\nHuman : Write an article in {style}")

            query = "insert into personality_templates values (%s, %s, %s, %s, %s);"

            run_query(query = query, values = tuple(values))

            logging.debug("Data is stored in your personality templates table")

        return content_
    

    def revise_content(self):
        '''To revise the generated content for simplification'''
        logging.info("Revising content")

        engine = get_engine()


        store = pd.read_sql_table("articles", con = engine)

        content_ = store[store["person"] == self.person].sort_values("created_at").iloc[-1, 2]
        article_id = int(store[store["person"] == self.person].sort_values("created_at").iloc[-1, 0])


        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system", "You are a AI agent whose role is to transform the provided content into a more concise, polished and engaging way"
            ),
            (
                "human", "Revise the following draft:\n\n{content}"
            )
        ]
        )

        template = prompt_template.invoke({"content":content_})

        if self.model_name == "GPT":

            model = ChatOpenAI(model="gpt-4o")

        elif self.model_name == "Ollama":

            model = ChatOllama(model="phi")

        elif self.model_name == "Gemini":

            model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=api)

        result = model.invoke(template)

        now = datetime.now()

        logging.debug("Content has been successfully revised")

        revised_store = pd.read_sql_table("revision_history", con = engine)
        
        revision_id = (
            int(revised_store["revision_id"].max()) + 1
            if not revised_store.empty and pd.notnull(revised_store["revision_id"].max())
            else 1
        )

        values = [
            revision_id,
            article_id,
            self.model_name,
            json.dumps(content_),
            json.dumps(sent_tokenize(result.content)),
            now.strftime("%Y-%m-%d %H:%M:%S"),
        ]

        insert_query = "insert into revision_history values (%s, %s, %s, %s, %s, %s);"

        run_query(query = insert_query, values = values)

        logging.debug("Data has been successfully stored in revision history table")

        revised_store = pd.read_sql_table("revision_history", con=engine)

        content = "".join(revised_store.sort_values("revised_at").iloc[-1, 4])

        print(content)

        meta_data = pd.read_sql_table("metadata", con = get_engine())

        values = []

        if meta_data["metadata_id"].shape[0] == 0:

            values.append(1)

        else:

            values.append(meta_data["metadata_id"].max() + 1)

        values.append(article_id)

        metacontent = [lemma.lemmatize(word) for word in word_tokenize(content) if word.lower() not in stp and not word.isdigit()]

        reading_time = round(len(word_tokenize(content)) / 200)

        publication_status = "Revised"

        values.extend([json.dumps(metacontent), reading_time, publication_status])

        query = "insert into metadata values (%s, %s, %s, %s, %s)"

        values = tuple(values)

        run_query(query = query, values = values)

        logging.debug("Revised content details are stored in metadata table")

        model = pd.read_sql_table("model_performance", con = get_engine())

        values = []

        if model["Test_id"].shape[0] == 0:

            values.append(1)

        else:
            values.append(model["Test_id"].max() + 1)

        values.extend([article_id, self.model_name])

        readability = textstat.flesch_reading_ease(content)

        score = round(readability / 100, 1)

        now = datetime.now()

        values.extend([now.strftime("%Y-%m-%d %H:%M:%S"), score, publication_status])

        query = "insert into model_performance values (%s, %s, %s, %s, %s, %s);"

        run_query(query = query, values = tuple(values))

        logging.debug("Revised data is stored in model performance table")

def recommendation_engine(user_query, k = 3):
    '''To recommend similar sentences to the user from the stored content based on the query'''

    logging.info(f"Fetching similar articles for query : {user_query}")

    similar_docs = {}

    query_embedding = embed_model.encode(user_query)

    results = collection.query(query_embeddings = [query_embedding],
                               n_results = k) # Uses cosine distance to calculate similarity
    
    logging.debug(f"Found {k} similar articles for the query : \"{user_query}\"")
    
    for similar, distance in zip(results["documents"][0], results["distances"][0]):

        similar = re.sub("[*#]", "", similar).strip()

        similar_docs[similar] = round(1 - distance, 2)

    return similar_docs.keys()