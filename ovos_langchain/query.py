import os

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from ovos_langchain.llm import get_llm


def get_local_qa(db, llm, hide_source=False):
    retriever = db.as_retriever()
    # activate/deactivate the streaming StdOut callback for LLMs
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not hide_source)
    return qa


if __name__ == "__main__":
    # example to query a local db
    load_dotenv()

    # Database settings
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
    persist_directory = os.environ.get('PERSIST_DIRECTORY')

    # generic model settings
    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')

    # gpt4all / llama specific settings
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # HuggingFace specific settings
    repo_id = os.environ.get("REPO_ID")  # replaces model_path

    # RWKV specific settings
    tokenizer_path = os.environ.get('TOKENIZER_PATH')
    strategy = os.environ.get('STRATEGY')

    # Define the Chroma settings
    CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

    mute_stream = False
    callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]
    llm = get_llm(model_type, callbacks, model_path, tokenizer_path, model_n_ctx, repo_id, strategy)

    qa = get_local_qa(db, llm)




    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        print("\n> Docs:")
        print([d.metadata for d in docs])
