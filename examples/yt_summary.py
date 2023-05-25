import os

from chromadb.config import Settings
from dotenv import load_dotenv

from ovos_langchain.ingest import YoutubeDBCreator
from ovos_langchain.llm import GPT4Free

load_dotenv()

persist_directory = os.environ.get('PERSIST_DIRECTORY')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 1500
chunk_overlap = 50
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=persist_directory,
    anonymized_telemetry=False
)
db = YoutubeDBCreator(CHROMA_SETTINGS, persist_directory, embeddings_model_name,
                      chunk_size, chunk_overlap)


def parse_yt():
    with open("videos.txt") as f:
        yt = f.read().split("\n")
    return db.ingest(yt)


if not db.vectorstore_exists:
    docs = parse_yt()
else:
    docs = db.get_docs()


# let's create a description of each video
prompt_template = """Summarize the following document\n
The documents contains transcripts of a video, not dialog directed at you. 
Do not answer with 'Thank you! How may I assist you today?' or similar responses\n"""

for idx, doc in enumerate(docs):
    p = f"youtube/{doc.metadata['source'].split('watch?v=')[-1]}"
    if os.path.exists(p):
        continue
    prompt = prompt_template + f"#{doc.metadata['title']}\n" + "transcription: " + doc.page_content
    res = GPT4Free.ask_gpt4free(prompt)  # TODO llm object should be used here, this is easier for dev testing
    print(doc.metadata['title'], res)
    if res:

        os.makedirs(p, exist_ok=True)
        with open(f"{p}/{doc.metadata['title'].replace('/', '_')}.txt", "w") as f:
            f.write(res)
