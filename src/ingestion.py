import os
from typing import List
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data_processing import process_documents

# Centralizar os metadados do projeto
FAISS_INDEX_PATH = "data/faiss_index"

def ingest_documents(raw_dir: str = "data/raw") -> FAISS:
    """
    Executa o pipeline completo de ingestão: 
    Extração -> Chunking -> Embeddings -> Vector Store.
    """
    load_dotenv()

    # 1. Processa e lê os documentos nativos preservando metadata
    docs = process_documents(raw_dir)
    if not docs:
        print("Nenhum documento para processar.")
        return None

    print("\nIniciando Segmentação (Chunking)...")
    
    # 2. Configura a heurística de Chunking
    # Parâmetros justificados (1000 chars com 200 de overlap): 
    # Ideal para textos de leis (artigos do CDC tendem a ser parágrafos longos,
    # overlap evita cortes no meio de um inciso ou cláusula vital).
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # 3. Mapeando e fragmentando cada doc adicionando ID do chunk
    all_chunks: List[Document] = []
    
    for doc in docs:
        # Segmenta o MD gigante gerado pelo docling
        chunks = text_splitter.split_text(doc.page_content)
        
        for i, text_chunk in enumerate(chunks):
            # O ID único exigido pelo projeto: chunk_id e doc_id
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_id"] = f"{chunk_metadata['doc_id']}-chunk-{i}"
            
            chunk_doc = Document(
                page_content=text_chunk,
                metadata=chunk_metadata
            )
            all_chunks.append(chunk_doc)
            
    print(f"Total de {len(all_chunks)} chunks gerados a partir de {len(docs)} documentos.")

    # 4. Geração de Embeddings Densa
    # Usando um modelo multilíngue do HuggingFace adequado pro Português BR
    print("\nInicializando o modelo de Embeddings (all-MiniLM-L6-v2 - HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Indexando chunks no banco vetorial FAISS...")
    vector_store = FAISS.from_documents(all_chunks, embeddings)

    # Salva o índice localmente para reutilização offline
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"Índice vetorial FAISS criado e salvo em: {FAISS_INDEX_PATH}")
    
    return vector_store


def get_retriever(k: int = 5):
    """
    Recupera o retriever (motor de busca) a partir do índice salvo.
    Se não encontrar o índice salvo, aborta e avisa o usuário.
    O 'k' representa o top-k a ser retornado pela busca KNN.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Índice FAISS não encontrado no caminho: {FAISS_INDEX_PATH}. Rode a ingestão primeiro.")
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # search_kwargs param define o "k" (quantidade de resultados)
    return vector_store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    import time
    start_time = time.time()
    ingest_documents()
    print(f"Tempo total de ingestão: {time.time() - start_time:.2f} segundos.")
