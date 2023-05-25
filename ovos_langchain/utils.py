import os

from chromadb.config import Settings
from ctransformers.langchain import CTransformers
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.llms import RWKV
from langchain.llms.base import LLM
from langchain.vectorstores import Chroma


class RWKVLLM(LLM):
    model: RWKV = None

    @classmethod
    def init(cls, model, tokens_path, strategy="cpu fp8"):
        cls.model = RWKV(model=model, strategy=strategy, tokens_path=tokens_path)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop=None,
            run_manager=None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

               # Instruction:
               {prompt}

               # Response:
               """
        return RWKVLLM.model(prompt)

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        return {}


def get_llm(model_type, callbacks, model_path, tokenizer_path, model_n_ctx, repo_id, strategy):
    # Prepare the LLM
    match model_type:
        case "HuggingFace":
            llm = HuggingFaceHub(repo_id=repo_id)
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks,
                           verbose=False, n_threads=16)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks,
                          verbose=False, n_threads=16)
        case "RWKV":
            RWKVLLM.init(model=model_path, tokens_path=tokenizer_path, strategy=strategy)
            llm = RWKVLLM()
        case "CTransformersMPT":
            llm = CTransformers(model=model_path, model_type='mpt')
        case "CTransformersGPT2":
            llm = CTransformers(model=model_path, model_type='gpt2')
        case "CTransformers":
            llm = CTransformers(model=model_path)
        case _default:
            print(f"Model {model_type} not supported!")
            exit(1)
    return llm


def get_RetrievalQA(db, model_type, model_path, tokenizer_path, model_n_ctx, repo_id, strategy,
                    mute_stream=True, hide_source=False):
    retriever = db.as_retriever()
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]
    llm = get_llm(model_type, callbacks, model_path, tokenizer_path, model_n_ctx, repo_id, strategy)
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
    qa = get_RetrievalQA(db, model_type, model_path, tokenizer_path, model_n_ctx, repo_id, strategy)

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
