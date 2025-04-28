import logging


logging.basicConfig(filename = "app.log", format = "%(asctime)s-%(levelname)s-%(module)s-%(lineno)d-%(message)s", datefmt = "%Y-%m-%d %H:%M:%S",level = logging.DEBUG)

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

logging.getLogger("SentenceTransformer").setLevel(logging.ERROR)

logging.getLogger("matplotlib").setLevel(logging.ERROR)

logging.getLogger("urllib3").setLevel(logging.ERROR)

logging.getLogger("posthog").disabled = True

logging.getLogger("PIL").setLevel(logging.ERROR)

logging.getLogger("chromadb.config").setLevel(logging.ERROR)

logging.getLogger("httpcore").setLevel(logging.ERROR)

logging.getLogger("httpx").setLevel(logging.ERROR)

logging.getLogger("uvicorn").setLevel(logging.ERROR)