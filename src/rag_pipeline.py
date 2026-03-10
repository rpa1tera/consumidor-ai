import os
from typing import List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.ingestion import get_retriever

load_dotenv()

class ConsumidorRAG:
    """
    Motor central RAG (Geração Aumentada por Recuperação) 
    para o Chatbot 'Consumidor.ai'.
    Integra a busca FAISS e Geração via Google Gemini com 
    ênfase na métrica de 'Groundedness' (Sem Alucinações).
    """
    def __init__(self, top_k: int = 5):
        # Configurar LLM (Google Gemini Flash-Lite é leve e adequado p/ CPU local requests via API)
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite", 
                temperature=0.1 # Temperatura baixa = Menos criatividade, mais aderência ao corpus
            )
        except Exception as e:
            raise RuntimeError("Erro ao carregar LLM. Verifique a chave 'GEMINI_API_KEY' no ambiente.") from e

        # Carrega o retriever salvado na indexação (baseline exige configurar top K).
        self.retriever = get_retriever(k=top_k)
        
        # Prompt Baseline Exigido pelo Projeto IFG: Groundedness e Recusa Estrita + Citações
        self.prompt = PromptTemplate.from_template(
            """Sua tarefa é atuar como um Assistente Especializado em Direitos do Consumidor ('Consumidor.ai').
Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE nos documentos delimitados pelo bloco <contexto> abaixo.

DIRETRIZES OBRIGATÓRIAS (GROUNDEDNESS E RECUSA):
1. Você não pode inventar, deduzir ou utilizar conhecimentos prévios externos ao contexto fornecido.
2. Se a resposta para a pergunta não estiver presente no contexto fornecido, você deve recusar explicitamente respondendo apenas: "Não encontrei essa informação na base de dados do Consumidor.Ai. Por favor, consulte um especialista ou órgão de proteção oficial."
3. Se a informação estiver no contexto, responda de forma clara e profissional.
4. Você DEVE incluir citações precisas sempre que afirmar algo, apontando o ID do trecho consultado no final da frase. Use o formato: [doc_id#chunk_id] extraído dos metadados fornecidos.

<contexto>
{context}
</contexto>

Pergunta: {question}
Resposta com citações:"""
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Formata os documentos recuperados na sintaxe esperada pelo prompt,
        comprimindo metadados para garantir que o LLM consiga citar a base corretamente.
        """
        formatted = []
        for i, doc in enumerate(docs):
            # Formatação essencial para orientar a citação pelo modelo
            doc_id = doc.metadata.get("doc_id", f"doc_{i}")
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
            fonte = doc.metadata.get("fonte", "Desconhecida")
            
            # Injetando uma "Header de Citação" dentro do próprio pedaço
            header = f"\n--- INÍCIO DO TRECHO [CITAÇÃO: {doc_id}#{chunk_id}] (Fonte: {fonte}) ---\n"
            content = doc.page_content
            footer = f"\n--- FIM DO TRECHO [CITAÇÃO: {doc_id}#{chunk_id}] ---\n"
            
            formatted.append(header + content + footer)
            
        return "\n".join(formatted)

    def query(self, question: str) -> dict:
        """
        Executa a query do usuário.
        Retorna tanto a resposta final do LLM quanto a lista de Documents fonte 
        recuperados (para auditoria e exibição na interface de transparência).
        """
        print(f"\n[RAG] Buscando contexto para a pergunta: '{question}'...")
        
        # 1. Recupera chunks
        source_docs = self.retriever.invoke(question)
        
        # Registo de Log (Obrigatório Pelo Baseline: "Expor via log chunks recuperados")
        print("\n--- LOG DE RECUPERAÇÃO DE CHUNKS ---")
        for d in source_docs:
             print(f"RECUPERADO: {d.metadata.get('doc_id')} | Chunk ID: {d.metadata.get('chunk_id')}")
        print("------------------------------------\n")

        # 2. Formata para injetar no Prompt
        formatted_context = self._format_docs(source_docs)

        # 3. Constroi a pipeline manual do LangChain (Chain)
        chain = self.prompt | self.llm | StrOutputParser()

        print("[RAG] Gerando resposta baseada em ancoragem estrita...")
        # 4. Invoca o modelo
        answer = chain.invoke({
            "context": formatted_context,
            "question": question
        })
        
        return {
            "answer": answer,
            "source_documents": source_docs
        }

if __name__ == "__main__":
    # Teste Rápido (Somente após a ingestão estar concluída)
    rag = ConsumidorRAG(top_k=3)
    response = rag.query("O que é considerado propaganda enganosa?")
    print("\nRESPOSTA RAG:")
    print(response["answer"])
