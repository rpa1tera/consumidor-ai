import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from src.rag_pipeline import ConsumidorRAG

# Tenta inicializar o Motor do Chatbot
try:
    chatbot_engine = ConsumidorRAG(top_k=5)
    READY = True
except Exception as e:
    chatbot_engine = None
    READY = False
    ERRO_MSG = str(e)

def chat_interaction(message, history):
    if not READY:
        return f"⚠️ Erro ao inicializar o Chatbot: {ERRO_MSG}"

    resposta_ditada = chatbot_engine.query(message)
    resposta_texto = resposta_ditada["answer"]
    documentos_fonte = resposta_ditada["source_documents"]

    # Painel de Transparência Estilizado com HTML para melhor legibilidade
    transparencia = "\n\n--- \n### 🔍 Fontes Consultadas\n"
    for idx, doc in enumerate(documentos_fonte, 1):
        trecho_curto = doc.page_content[:150].replace("\n", " ") + "..."
        doc_id = doc.metadata.get('doc_id', 'N/A')
        transparencia += f"**{idx}.** *{trecho_curto}* (ID: `{doc_id}`)\n"

    return resposta_texto + transparencia

# --- UI DESIGN (STANDARD GRADIO) ---
with gr.Blocks(title="Consumidor.ai", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ⚖️ Consumidor.Ai
        ### Seu assistente jurídico especializado em Direitos do Consumidor
        
        > 🤖 **Transparência**: Baseado estritamente no Código de Defesa do Consumidor e cartilhas do PROCON. Todas as respostas possuem citações e fontes auditáveis em conformidade com as exigências RAG (Baseline).
        """
    )
    
    chat = gr.ChatInterface(
        fn=chat_interaction,
        chatbot=gr.Chatbot(
            show_label=False,
            height=550,
            avatar_images=(None, "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ef629256cf65f.svg") # Ícone opcional
        ),
        textbox=gr.Textbox(
            placeholder="Digite sua dúvida aqui (ex: O que fazer se recebi um produto com defeito?)...",
            container=False,
            scale=7
        ),
        submit_btn=gr.Button("Enviar", variant="primary"),
        examples=[
            "Quais são meus direitos em caso de atraso de voo?",
            "Como cancelar um contrato de internet sem multa?",
            "Comprei um produto online e me arrependi. O que devo fazer?"
        ]
    )

if __name__ == "__main__":
    print("\n[Consumidor.ai] Interface Gradio Padrão carregada com sucesso.")
    demo.launch(share=False)