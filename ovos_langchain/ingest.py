#!/usr/bin/env python3
import glob
import os
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import (
    UnstructuredURLLoader,
    YoutubeLoader
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownTextSplitter, NLTKTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm
from langchain.schema import BaseDocumentTransformer, Document
from ovos_langchain.loaders import LOADER_MAPPING, RssReader


@dataclass
class DBCreator:
    """ generic DB creator ingesting all kinds of files """
    db_settings: Settings
    persist_directory: str
    embeddings_model_name: str
    chunk_size: int = 1000
    chunk_overlap: int = 50
    ignored_files: List[str] = None

    @property
    def vectorstore_exists(self) -> bool:
        """
        Checks if vectorstore exists
        """
        if os.path.exists(os.path.join(self.persist_directory, 'index')):
            if os.path.exists(os.path.join(self.persist_directory, 'chroma-collections.parquet')) and os.path.exists(
                    os.path.join(self.persist_directory, 'chroma-embeddings.parquet')):
                list_index_files = glob.glob(os.path.join(self.persist_directory, 'index/*.bin'))
                list_index_files += glob.glob(os.path.join(self.persist_directory, 'index/*.pkl'))
                # At least 3 documents are needed in a working vectorstore
                if len(list_index_files) > 3:
                    return True
        return False

    def get_docs(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)

        db = Chroma(persist_directory=self.persist_directory,
                    embedding_function=embeddings, client_settings=self.db_settings)
        do = db.get()

        return [Document(page_content=txt, metadata=meta)
                for meta, txt in zip(do["metadatas"], do["documents"])]

    def ingest(self, documents):
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)

        db = Chroma(persist_directory=self.persist_directory,
                    embedding_function=embeddings, client_settings=self.db_settings)

        if self.vectorstore_exists:
            # Update and store locally vectorstore
            print(f"Appending to existing vectorstore at {persist_directory}")
            collection = db.get()
            self.ignored_files = [metadata['source'] for metadata in collection['metadatas']]
        else:
            self.ignored_files = []

        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(documents)

        db.persist()
        db = None

        print(f"Ingestion complete! You can now query your documents")
        return documents



@dataclass
class FileDBCreator(DBCreator):
    """ generic DB creator ingesting all kinds of files """

    @staticmethod
    def load_single_document(file_path: str) -> Document:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()[0]

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_markdown(self, path) -> List[Document]:
        """
        Loads all markdown documents from the source documents directory, ignoring specified files
        """
        all_files = []
        for root, folders, files in os.walk(path):
            all_files += [f"{root}/{f}" for f in files if f.endswith(".md")]
        return self.load_files(all_files)

    def load_files(self, file_paths) -> List[Document]:
        """
        Loads all documents from the source documents directory, ignoring specified files
        """
        self.ignored_files = self.ignored_files or []
        filtered_files = [file_path for file_path in file_paths
                          if file_path not in self.ignored_files]

        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
                for i, doc in enumerate(pool.imap_unordered(self.load_single_document, filtered_files)):
                    results.append(doc)
                    pbar.update()

        return results

    def process_documents(self, path, split = False) -> List[Document]:
        """
        Load documents and split in chunks
        """
        print(f"Loading documents from {path}")
        documents = self.load_markdown(path)
        if not documents:
            print("No new documents to load")
            return []
        print(f"Loaded {len(documents)} new documents from {path}")
        if not split:
            return documents
        # text_splitter = NLTKTextSplitter(chunk_size=self.chunk_size)
        text_splitter = MarkdownTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {self.chunk_size} tokens each)")
        return texts

    @property
    def vectorstore_exists(self) -> bool:
        """
        Checks if vectorstore exists
        """
        if os.path.exists(os.path.join(self.persist_directory, 'index')):
            if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(
                    os.path.join(self.persist_directory, 'chroma-embeddings.parquet')):
                list_index_files = glob.glob(os.path.join(self.persist_directory, 'index/*.bin'))
                list_index_files += glob.glob(os.path.join(self.persist_directory, 'index/*.pkl'))
                # At least 3 documents are needed in a working vectorstore
                if len(list_index_files) > 3:
                    return True
        return False

    def ingest(self, path):
        texts = self.process_documents(path)
        return super().ingest(texts)



@dataclass
class YoutubeDBCreator(DBCreator):

    def process_youtube(self, urls):
        documents = []
        for url in urls:
            print(f"parsing video metadata: {url}")
            try:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, continue_on_failure=True)
                doc = loader.load()
                if doc:
                    doc = doc[0]
                    doc.metadata = {k: v if isinstance(v, (str, int, float)) else str(v)
                                    for k, v in doc.metadata.items() if
                                    v is not None}  # to avoid bug if metadata contains None values
                    documents.append(doc)
            except Exception as e:
                print(e)
        print(f"Loaded {len(documents)} new video info from {len(urls)} youtube urls")
        return documents

    def ingest(self, urls=None):
        texts = self.process_youtube(urls)
        return super().ingest(texts)



@dataclass
class WebsiteDBCreator(YoutubeDBCreator):

    def process_urls(self, urls, split = False):
        yt = [u for u in urls if "youtube." in u]
        urls = [u for u in urls if u not in yt]
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        print(f"Loaded {len(documents)} new documents from {len(urls)} urls")
        if split:
            text_splitter = NLTKTextSplitter(chunk_size=self.chunk_size)
            texts = text_splitter.split_documents(documents)
            print(f"Split into {len(texts)} chunks of text (max. {self.chunk_size} tokens each)")
            if yt:
                texts += self.process_youtube(yt)
            return texts
        if yt:
            documents += self.process_youtube(yt)
        return documents

    def process_rss(self, urls, split = False):
        reader = RssReader()
        documents = reader.load_data(urls)
        purls = set([doc.extra_info.get("link") for doc in documents if doc.extra_info.get("link")])
        print(f"Loaded {len(purls)} new urls from {len(urls)} rss feeds")
        return self.process_urls(purls, split)

    def ingest(self, urls=None, rss_urls=None):
        texts = []
        if urls:
            texts += self.process_urls(urls)

        if rss_urls:
            texts += self.process_rss(rss_urls)

        return super().ingest(texts)


if __name__ == "__main__":
    load_dotenv()

    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    persist_directory = "ovos_website_index"
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
    chunk_size = 1500
    chunk_overlap = 50

    # Define the Chroma settings
    CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )

    urls = [
        "https://openvoiceos.com",
        "https://www.gofundme.com/f/openvoiceos",
        "https://blog.graywind.org/posts/naptime-skill",
        "https://blog.graywind.org/posts/coqui-tts-neon-ovos",
        "https://blog.graywind.org/posts/neon-custom-wakeword",
        "https://blog.graywind.org/posts/neon-change-wakeword"
    ]

    rss = [
        "https://mycroft.ai/feed",
        "https://www.reddit.com/r/OpenVoiceOS.rss",
        "https://www.reddit.com/r/MycroftAi.rss"
    ]

    db = WebsiteDBCreator(CHROMA_SETTINGS, persist_directory,
                          embeddings_model_name, chunk_size, chunk_overlap)
    db.ingest(urls, rss)
