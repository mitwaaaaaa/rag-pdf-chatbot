#create vectors
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os


load_dotenv()

hf_client = InferenceClient(token=os.getenv("HF_API_KEY"))
reader = PDFReader()
# use llamaindex to read the pdf then split sentences from that pdf into chunks-> embed them -> store in vector db
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384 #should match what we defined in vector db
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
#how much of 1 chunk is overlapping into the beginning of next chunk
# hello world my name is Mitwa, Chunks overlap=1: Hello world my, my name is Mitwa (so we don't lose relevant context)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path) #look for pdf and load it
    texts = [d.text for d in docs if getattr(d, "text", None)] #We'll get all the text content for every single doc in docs
    #if the document has some "text" attribute
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

#send a req to the AI model and pass all of the text which is already chunked
# it will convert it into embedding (vector) which we can store in db
def embed_texts(texts):
    return hf_client.feature_extraction(
        texts,
        model=EMBED_MODEL
    )
#we are gonna pull out the embeddings
