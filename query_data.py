import argparse
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv

load_dotenv()

print(os.environ.get('OPENAI_KEY'))

openaikey = os.environ.get('OPENAI_KEY')

print(openaikey)

CHROMA_PATH="chroma"

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

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {query}
"""

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('query_text', type=str, help="The query text.")
  args = parser.parse_args()
  query_text = args.query_text

  embedding_function = OpenAIEmbeddings(openai_api_key=openaikey)

  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

  results = db.similarity_search_with_relevance_scores(query_text, k=3)

  if len(results) == 0 or results[0][1] < 0.7:
    print("unable to find matching results")
    return

  context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, query=query_text)
  print(prompt)

  model = ChatOpenAI(openai_api_key=openaikey)
  response_text = model.invoke(prompt)

  sources = [doc.metadata.get("source", None) for doc, _score in results]
  formatted_response = f"Response :{response_text}\nSources: {sources}"
  print(formatted_response)



if __name__ == "__main__":
  main()