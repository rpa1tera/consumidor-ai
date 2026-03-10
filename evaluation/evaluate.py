import sys
import os
import json

# Adiciona a pasta raiz ao Python path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import get_retriever

def load_golden_set(filepath="evaluation/golden_set.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_retriever_evaluation(k_valores=[3, 5, 10]):
    """
    Avalia a etapa de RECALL da busca vetorial (FAISS).
    Testa se pelo menos uma referência da Resposta Esperada
    está contida no top-k recuperado. Retorna Recall Rate.
    """
    golden_set = load_golden_set()
    resultados = {k: 0 for k in k_valores}
    perguntas_no_corpus = [q for q in golden_set if q["tipo"] != "Fora do Corpus"]
    total_perguntas = len(perguntas_no_corpus)
    
    print(f"Iniciando Avaliação do Retriever (Apenas perguntas válidas do Corpus: {total_perguntas} itens).")

    for k in k_valores:
        # Carrega retriever com o K específico
        retriever = get_retriever(k=k)
        acertos = 0
        
        for item in perguntas_no_corpus:
            docs_recuperados = retriever.invoke(item["pergunta"])
            textos_brutos = " ".join([d.page_content.lower() for d in docs_recuperados])
            
            # Aqui fazemos um match elástico (palavras chaves ou referências)
            # Para avaliação cega exata, seria usado LLM for metrics (ex: RAGAS), 
            # mas vamos avaliar o baseline comparativo.
            ref_esperada = item["referencia_esperada"].lower().replace(".", "").split(" ")
            
            # Checa se pedaços vitais da referência/texto esperado aparecem no corpo recuperado
            match = any(word in textos_brutos for word in ref_esperada if len(word) > 4)
            if match:
                acertos += 1
                
        resultados[k] = (acertos / total_perguntas) * 100
        print(f"Recall@{k}: {resultados[k]:.2f}% ({acertos}/{total_perguntas})")

    return resultados

if __name__ == "__main__":
    print("--- AVALIAÇÃO DO MOTOR RAG BASELINE ---")
    run_retriever_evaluation()
