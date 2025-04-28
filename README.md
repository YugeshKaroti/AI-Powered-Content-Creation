# AI-Powered-Content-Creation and Recommendation Platform

## Project Overview

This project aims to empower users to generate high-quality articles using multiple Large Language Models (LLMs) and delivers personalized reading recommendations. It combines AI-driven content creation with semantic search, reader behavior tracking, and content revision workflows, providing an end-to-end solution for content platforms.

## Key Features
1) Content Creation Module : Select from different LLMs (GPT-4, Gemini, Ollama) using a Model Switchboard. Generate articles based on personality-driven templates (e.g., Technical Writers, Sci-Fi Authors). Save original content along with topic, persona, style, and model used.

2) Content Revision Engine : Revise articles to make them more concise, polished, and engaging. Track revision history with timestamped entries.

3) Recommendation System : Vectorize content using SentenceTransformer embeddings. Store embeddings in ChromaDB for fast semantic similarity searches. Recommend similar articles based on content embeddings.

4) Database Architecture : Structured storage using MySQL by defining tables like Articles, Revision History, Metadata, Model Performance, Personality Templates, Sentence Embeddings.

5) Reader Experience : Provides clean Streamlit interface for browsing articles. Smart sidebar showing Top 5 recommended articles. Clickable links to generate content and recommend similar sentences.

## Used Technologies 
1) Frontend : Streamlit

2) Backend : Python, Langchain, LLM APIs

3) Vector Database : ChromaDB

4) LLMs : OpenAI (GPT-4o), Gemini (Gemini-1.5-pro), Ollama (phi)

5) Embeddings : SentenceTransformers (all-MiniLM-L6-v2)

6) Database : MySQL

7) Visualization : Matplotlib, WordCloud

8) Other : dotenv, NLTK, Pickle

## Outcome

1) Speeds up the content creation and publication cycle through AI assistance.

2) Improves reader engagement with smart recommendations.

3) Provides a modular, extensible framework for integrating additional LLMs and recommendation strategies.