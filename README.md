<h1 align="center">
  ⚖️ Consumidor.Ai
</h1>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white">
  <img alt="RAG" src="https://img.shields.io/badge/Architecture-RAG-orange">
  <img alt="Gradio" src="https://img.shields.io/badge/Interface-Gradio-ff69b4?logo=gradio&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/Framework-LangChain-green?logo=langchain&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

<p align="center">
  <strong>Um assistente inteligente (RAG) especializado em Direito do Consumidor, capaz de processar documentos extensos e responder dúvidas com base na legislação e jurisprudência.</strong>
</p>

<hr>

## 🚀 Visão Geral

O **Consumidor.Ai** é um chatbot baseado na arquitetura **RAG (Retrieval-Augmented Generation)**. Ele ingere documentos jurídicos (como o Código de Defesa do Consumidor, jurisprudências e processos), processa esses textos e utiliza modelos de linguagem avançados (LLMs) para gerar respostas precisas e fundamentadas.

O foco principal deste projeto é lidar de forma eficiente com PDFs extensos, superando limitações de memória ao realizar o *chunking* inteligente de documentos antes da extração de dados.

## 🧠 Arquitetura e Tecnologias

O projeto foi construído utilizando as ferramentas mais modernas do ecossistema de dados e IA em Python:

- **Extração de Dados:** `PyMuPDF` (divisão inteligente de PDFs) e `Docling` (parseamento avançado de layout e tabelas).
- **Processamento de Linguagem Natural (NLP):** `LangChain`.
- **Embeddings:** Modelos da família `Sentence-Transformers` (`HuggingFace`).
- **Vector Database:** `FAISS` (armazenamento e busca de similaridade vetorial otimizados para CPU).
- **LLM / Geração:** Integração com APIs de LLMs (como Google GenAI).
- **Interface Gráfica:** `Gradio`, oferecendo um chat interativo e fácil de usar.
- **Gerenciamento de Pacotes:** `uv` via `pyproject.toml`.

## 📂 Estrutura do Projeto

A organização segue boas práticas de engenharia de software e ciência de dados:

```text
consumidor-ai/
├── app/
│   └── main.py              # Ponto de entrada da interface Gradio (Chatbot UI)
├── data/
│   ├── raw/                 # PDFs e documentos brutos originais
│   └── faiss_index/         # Banco de dados vetorial FAISS após a ingestão
├── docs/                    # Documentação adicional do projeto
├── evaluation/              # Scripts e relatórios de avaliação de baseline
├── notebooks/               # Jupyter Notebooks para experimentação e testes (01_rename_jsons, etc)
├── prompts/                 # Templates de prompts do LangChain para o LLM
├── src/                     # Código-fonte principal (lógica de negócios)
│   ├── data_processing.py   # Lógica de chunking de PDFs grandes e parseamento inicial
│   ├── ingestion.py         # Uso do Docling para extrair texto/tabelas e salvar metadados
│   └── rag_pipeline.py      # Criação do FAISS, Retriever e cadeia gerativa (Chain)
├── .env                     # Variáveis de ambiente (Chaves de API) - [NÃO COMMITAR]
├── .gitignore               # Ignora arquivos sensíveis, caches e dados grandes
├── pyproject.toml           # Configuração do projeto e dependências (uv/pip)
├── README.md                # Este arquivo
```

## ⚙️ Como Instalar e Rodar

### 1. Pré-requisitos
Certifique-se de ter o **Python 3.12 ou superior** instalado. Recomenda-se o uso do `uv` para instalar dependências rapidamente.

### 2. Clonar o Repositório
```bash
git clone https://github.com/rpa1tera/consumidor-ai.git
cd consumidor-ai
```

### 3. Configurar Ambiente Virtual e Dependências
Crie um ambiente virtual e instale as bibliotecas definidas no `pyproject.toml`:
```bash
# Se usar UV (Recomendado)
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
uv pip sync pyproject.toml

# Ou usando PIP padrão:
python -m venv .venv
# ative o ambiente virtual
pip install -r requirements.txt # se houver, ou pip install .
```

### 4. Configurar as Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto (como o exemplo abaixo) e insira suas chaves de API necessárias (ex: Google Gemini, OpenAI, etc., dependendo de qual LLM você configurou no `rag_pipeline.py`):

```env
GOOGLE_API_KEY="sua_chave_aqui"
# Outras chaves necessárias
```

### 5. Ingerir os Documentos (Construir o Banco de Dados)
Coloque seus arquivos PDF na pasta `data/raw/` e execute o script de ingestão para processar os textos e gerar o índice FAISS:

```bash
python src/ingestion.py
```
*Isso criará a estrutura de vetores dentro da pasta `data/faiss_index/`.*

### 6. Iniciar o Chatbot
Inicie a interface do Gradio para conversar com a Inteligência Artificial:

```bash
python app/main.py
```
Acesse no seu navegador o endereço gerado no terminal (geralmente `http://127.0.0.1:7860`).

<hr>

## 🛠️ Contribuição e Problemas Comuns

- **Out of Memory (OOM) com Docling:** Este projeto implementa uma pipeline robusta no `data_processing.py` para pré-fatiar (`chunk`) PDFs massivos usando `PyMuPDF` antes de enviá-los ao `Docling`, prevenindo travamentos em máquinas com pouca RAM.
- Para sugerir melhorias, sinta-se à vontade para abrir uma *Issue* ou submeter um *Pull Request*.

---
<p align="center">
  Desenvolvido por <strong>Camilla Rodrigues e Raquel Alcântara</strong> na Pós-Graduação em IA Aplicada - IFG.
</p>
