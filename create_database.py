from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os
import shutil

from dotenv import load_dotenv

load_dotenv()

print(os.environ.get('OPENAI_KEY'))

openaikey = os.environ.get('OPENAI_KEY')


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

BASE_DIR="/"

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

CHROMA_PATH = "chroma"

DATA_PATH="data/books"

def main():
    generate_data_store()

def generate_data_store():
  documents = load_documents()
  chunks = split_text(documents)
  save_to_chroma(chunks)

def load_documents():
  loader = DirectoryLoader(DATA_PATH, glob="*.md")
  documents = loader.load()
  return documents

def split_text(documents: list[Document]):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True
  )
  chunks = text_splitter.split_documents(documents)
  print(f"Split {len(documents)} documents into {len(chunks)} chunks")
  document = chunks[10]
  print(document.page_content)
  print(document.metadata)

  return chunks

def save_to_chroma(chunks: list[Document]):
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

  db = Chroma.from_documents(
    chunks, OpenAIEmbeddings(openai_api_key=openaikey), persist_directory=CHROMA_PATH
  )

  db.persist()

  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "__main__":
  main()







