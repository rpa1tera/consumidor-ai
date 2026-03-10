# Análise Crítica do Baseline e Melhorias

## Trade-offs Adotados no Baseline
1. **Memória vs Performance (Docling):** O Docling extrai Layout e OCR via Machine Learning, o que provou ser insustentável na CPU do Desktop hospedeiro ao processar calhamaços do PROCON de uma só vez (causando Out of Memory/ std::bad_alloc). **Ação corretiva:** A técnica de fragmentação (PyMuPDF) em 15 páginas, limitou os picos de consumo criando estabilidade, mas estendeu um pouco o tempo total sistêmico da indexação em disco.
2. **Avaliação Primitiva via String Match:** O Recall das 20 perguntas baseou-se em busca de Palavras Chaves dentro dos fragmentos do FAISS. O RAG trabalha por vizinhança semântica que uma simples checagem textual ignora, explicando métricas como *Recall@10 (33.33%)*. 

## Trilhas de Melhoria Promissoras (Próximos Passos)
Segundo o repositório da disciplina, após consolidar este baseline funcional de Citações Restritas, devemos mirar em:

- **Trilha 6 (Avançada de Recuperação / MMR):** Aplicar Recuperação de Relevância Marginal Máxima em vez da Similaridade de Cossenos base para remover redundância entre os Chunks.
- **Trilha 4 (Engenharia de Prompt Expandida):** Como as leis do CDC alteram conceitos de ônus e reparação, o prompt principal pode incorporar um sub-agente (Chain of Thought/Step-Back Prompting) para decompor as perguntas.
- **Implementação do RAGAS (RAG Assessment):**  Substituir a atual correspondência estrita (*evaluate.py*) pelos módulos contextuais do RAGAS que usariam uma instância do Gemini para testar o Groundedness de forma probabilística e avaliar fidedignidade real do pipeline.
