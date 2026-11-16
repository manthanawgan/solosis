from ingestion import ingestion
from embeddings import store_embeddings
from 


memory_candidates = ingestion(file_path)

extracted_results = extract_memories(memory_candidates)

embeddings = generate_embeddings(extracted_results)

store_embeddings(extracted_results, embeddings)