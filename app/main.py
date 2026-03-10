import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from src.rag_pipeline import ConsumidorRAG

# Tenta inicializar o Motor do Chatbot, dependente do FAISS carregado.
# Como k=5 na especificação, o FAISS buscará o Top-5.
try:
    chatbot_engine = ConsumidorRAG(top_k=5)
    READY = True
except Exception as e:
    chatbot_engine = None
    READY = False
    ERRO_MSG = str(e)


def chat_interaction(message, history):
    """
    Função principal que o ChatInterface do Gradio chamará.
    Ela processa a query e formata um Log de transparência contendo 
    exatamente os trechos recuperados do FAISS (Requiremento Mínimo).
    """
    if not READY:
        return f"⚠️ Erro ao inicializar o Chatbot: {ERRO_MSG}\n\nCertifique-se de que o script `src/ingestion.py` foi rodado para criar o corpus."

    # Interroga o RAG
    resposta_ditada = chatbot_engine.query(message)
    resposta_texto = resposta_ditada["answer"]
    documentos_fonte = resposta_ditada["source_documents"]

    # Constrói o Painel de Transparência (Documentos Recuperados)
    transparencia = "\n\n---\n**🔍 Transparência RAG (Documentos Recuperados para fundamentação):**\n"
    for idx, doc in enumerate(documentos_fonte, 1):
        # Limita visualização para não poluir muito a UI (Primeiros 150 caracteres)
        trecho_curto = doc.page_content[:150].replace("\n", " ") + "..."
        doc_id = doc.metadata.get('doc_id')
        chunk_id = doc.metadata.get('chunk_id')
        transparencia += f"- **Ref {idx}** (`{doc_id}#{chunk_id}`): _{trecho_curto}_\n"

    # Junta a resposta gerada com a barra de transparência
    resposta_final = resposta_texto + transparencia
    return resposta_final


# Construção da Interface
with gr.Blocks() as demo:
    gr.Markdown("# ⚖️ Consumidor.Ai")
    gr.Markdown("### Seu assistente especializado em Direitos do Consumidor")
    gr.Markdown("> 🤖 **Transparência**: Baseado estritamente no Código de Defesa do Consumidor e cartilhas do PROCON. Todas as respostas possuem citações e fontes auditáveis em conformidade com as exigências RAG (Baseline).")
    
    chat = gr.ChatInterface(
        fn=chat_interaction,
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Faça sua pergunta sobre direito do consumidor...", container=False, scale=7),
        examples=[
            "Comprei um produto online e me arrependi. O que devo fazer?",
            "O que é venda casada?",
            "Como funciona a garantia legal e contratual?",
            "Posso cancelar a matrícula da escola?" # Pergunta possivelmente OOC dependendo do PROCON
        ],
    )

if __name__ == "__main__":
    print("\n[Consumidor.ai] Iniciando Servidor Gradio...")
    demo.launch(share=False, theme=gr.themes.Soft())
