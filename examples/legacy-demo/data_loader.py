from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(file_path: str):
    reader = PDFReader()
    documents = reader.load_data(file=file_path)
    texts = [d.text for d in documents if getattr(d, 'text', None)]
    chunked_texts = []
    for text in texts:
        chunks = splitter.split_text(text)
        chunked_texts.extend(chunks)
    return chunked_texts

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
    )
    return [item.embedding for item in response.data]