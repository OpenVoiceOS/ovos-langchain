from llama_index import GPTVectorStoreIndex, download_loader
import os
import random
import uuid
from threading import Event
from typing import Any

from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferMemory
from ovos_langchain.llm import get_llm

# example to query a local db
load_dotenv()

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

mute_stream = False
callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]
llm = get_llm(model_type, callbacks, model_path, tokenizer_path, model_n_ctx, repo_id, strategy)


RedditReader = download_loader('RedditReader')

subreddits = ['MachineLearning']
search_keys = ['PyTorch', 'deploy']
post_limit = 10

loader = RedditReader()
documents = loader.load_data(subreddits=subreddits, search_keys=search_keys, post_limit=post_limit)
index = GPTVectorStoreIndex.from_documents(documents)

tools = [
    Tool(
        name="Reddit Index",
        func=lambda q: index.query(q),
        description=f"Useful when you want to read relevant posts and top-level comments in subreddits.",
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", memory=memory
)

output = agent_chain.run(input="What are the pain points of PyTorch users?")
print(output)
