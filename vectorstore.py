import json
from typing import List
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

def chunk_text(document_text: str, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(document_text)

def get_ollama_embedding(text: str, model_name: str = "mxbai-embed-large") -> List[float]:
    """
    Calls Ollama's native Python embeddings function for the given text,
    returning a list of floats as the embedding vector.
    """
    # You can modify the prompt as desired. 
    # The official example is something like:
    # "Represent this sentence for searching relevant passages: ...".
    prompt = f"Represent this sentence for searching relevant passages: {text}"
    # print('embedding start')
    # print(prompt)

    try:
        # Use Ollama's embeddings() function
        result = ollama.embeddings(
            model="mxbai-embed-large",
            prompt=prompt
        )
        # 'result' is typically a dict with an "embedding" key
        embedding = result.get("embedding", [])
        # print('embedding done')
        return embedding
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return []

def store_document_in_qdrant(document_text: str, collection_name="my_collection"):
    """
    - Splits the document into chunks
    - Embeds each chunk with Ollama (mxbai-embed-large)
    - Stores/upserts the result in Qdrant
    """
    client = QdrantClient(url="http://localhost:6333")

    # 1. Determine embedding dimension by running a sample
    sample_embed = get_ollama_embedding("Hello world", "mxbai-embed-large")
    # print(sample_embed)
    vector_size = len(sample_embed)

    # 2. Create or update Qdrant collection
    #    (This version only re-creates if it doesn't exist,
    #    or if dimension is mismatched. Adjust as you like.)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    print(f"Creating collection '{collection_name}' with dimension {vector_size}.")


    # try:
    #     collection_info = client.get_collection(collection_name=collection_name)
    #     existing_config = collection_info.dict().get("config", {})
    #     existing_vectors_config = existing_config.get("params", {}).get("vectors", {})
    #     existing_dim = existing_vectors_config.get("size")

    #     if existing_dim and existing_dim != vector_size:
    #         print(f"WARNING: Existing collection dimension {existing_dim} != {vector_size}. Recreating.")
    #         client.recreate_collection(
    #             collection_name=collection_name,
    #             vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    #         )
    #     else:
    #         print(f"Collection '{collection_name}' already exists with matching dimension. Upserting data...")

    # except Exception:
    #     # If the collection doesn't exist, create it:
    #     print(f"Creating collection '{collection_name}' with dimension {vector_size}.")
    #     client.create_collection(
    #         collection_name=collection_name,
    #         vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    #     )

    # 3. Chunk the document text
    chunks = chunk_text(document_text)
    print(f"Split the document into {len(chunks)} chunks.")
    # 4. Embed each chunk and prepare Qdrant points
    points = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        embedding = get_ollama_embedding(chunk, "mxbai-embed-large")
        points.append(
            PointStruct(
                id=idx, 
                vector=embedding,
                payload={"text": chunk}
            )
        )

    # 5. Upsert to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"Stored {len(chunks)} chunks in collection '{collection_name}'.")


def get_all_points(collection_name="my_collection"):
    """
    Scroll through the collection, return a list of text chunks
    from the 'text' field in each record's payload.
    """
    client = QdrantClient(url="http://localhost:6333")

    all_points = []
    offset = None

    while True:
        # 'client.scroll' returns a tuple: (points_batch, next_offset)
        points_batch, next_offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=50,
            with_payload=True
        )

        if not points_batch:
            # No more records in this batch, stop
            break

        # Accumulate the new records
        all_points.extend(points_batch)

        if next_offset is None:
            # No more pages to scroll
            break

        # Update offset for the next iteration
        offset = next_offset

    # Extract 'text' field from each record's payload
    chunks = []
    for record in all_points:
        chunk_text = record.payload.get("text", "")
        chunks.append(chunk_text)

    return chunks