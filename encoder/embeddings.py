from typing import List
import numpy as np

from fastembed import TextEmbedding

from qdrant_client import QdrantClient, models


embedding_model = TextEmbedding(
    model_name = "BAAI/bge-small-en"
)

def embedding(extracted_results):

    # documents: List[str]     

    # for memory in extracted_results.memories:
    #     documents.append(memory.content)

    documents: List[str] = [
        memory.content for memory in extracted_results.memories
    ]

    if not documents:
        return []

    embedding_generator = embedding_model.embed(documents)

    embedding_list = list(embedding_generator)

    return embedding_list



client = QdrantClient(path="./qdrant_data")

client.create_collection(
    collection_name = "agent_memories",
    vectors_config = models.VectorParams(
        size = client.get_embedding_size(embedding_model),
        distance = models.Distance.COSINE
    )
)

def store_embeddings(extracted_results, embedding_list):

    points = []

    for memory, vector in zip(
        extracted_results.memories,
        embedding_list
    ):
        points.append(
            models.PointStruct(
                id = memory.id,
                vector = vector.tolist(),
                payload = {
                    "content": memory.content,
                    "category": memory.category,
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "source_message_ids": memory.source_message_ids
                }
            )
        )

    client.upsets(
        collection_name = "agent_memories",
        points = points
    )


    