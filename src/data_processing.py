import os
import tempfile
from pathlib import Path
from typing import List
import pymupdf # PyMuPDF
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

def process_documents(raw_dir: str = "data/raw") -> List[Document]:
    """
    Processa arquivos PDF/HTML usando Docling. 
    Para evitar 'Out of Memory' em PDFs grandes, fatiamos usando PyMuPDF
    antes de jogar para o motor primário (Docling).
    """
    
    print("\n[DOCLING] Inicializando modo de leitura. OCR profundo desabilitado.")
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,           
        do_table_structure=False 
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.HTML],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    documents = []
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        print(f"Diretório {raw_dir} não encontrado.")
        return documents

    supported_extensions = [".pdf", ".html"]
    
    print(f"Iniciando conversão de documentos em {raw_dir}...")
    
    for file_path in raw_path.iterdir():
        if file_path.suffix.lower() == ".html":
            print(f"  -> Converte: {file_path.name}")
            try:
                result = converter.convert(str(file_path))
                text_content = result.document.export_to_markdown()
                doc_id = file_path.stem.replace(" ", "_").lower()
                metadata = {
                    "doc_id": doc_id,
                    "titulo": file_path.stem.replace("_", " ").title(),
                    "fonte": file_path.name,
                    "tipo": "HTML"
                }
                documents.append(Document(page_content=text_content, metadata=metadata))
            except Exception as e:
                print(f"     ❌ Erro Crítico HTML: {e}")
                
        elif file_path.suffix.lower() == ".pdf":
            print(f"  -> Converte fatiado: {file_path.name}")
            try:
                # Fatiar com PyMuPDF para evitar std::bad_alloc do Docling (C++)
                pdf_doc = pymupdf.open(str(file_path))
                total_pages = len(pdf_doc)
                chunk_size = 15 # Lote de páginas seguras para 8GB RAM
                
                doc_id = file_path.stem.replace(" ", "_").lower()
                titulo = file_path.stem.replace("_", " ").title()
                
                for start_page in range(0, total_pages, chunk_size):
                    end_page = min(start_page + chunk_size, total_pages) - 1
                    
                    # Arquivo Temporário
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                        temp_path = temp_pdf.name
                    
                    # Exportar fatia do PyMuPDF
                    pdf_slice = pymupdf.Document()
                    pdf_slice.insert_pdf(pdf_doc, from_page=start_page, to_page=end_page)
                    pdf_slice.save(temp_path)
                    pdf_slice.close()
                    
                    print(f"     -> Extraindo via DOCLING páginas {start_page+1} a {end_page+1} de {total_pages}...")
                    
                    # Converte usando exclusivamente o DOCLING
                    result = converter.convert(temp_path)
                    text_content = result.document.export_to_markdown()
                    
                    metadata = {
                        "doc_id": doc_id,
                        "titulo": titulo,
                        "fonte": file_path.name,
                        "tipo": "PDF",
                        "lote": f"pags_{start_page+1}_a_{end_page+1}"
                    }
                    
                    # Armazena na pool do Langchain
                    documents.append(Document(page_content=text_content, metadata=metadata))
                    
                    # Limpeza Imediata da fatia pra salvar disco/memória
                    os.remove(temp_path)
                    
                pdf_doc.close()
                print(f"     ✅ Sucesso na extração global do documento.")
            except Exception as e:
                print(f"     ❌ Erro Crítico ao processar fatia do PDF: {e}")

    print(f"\nTotal de lotes convertidos para Langchain: {len(documents)}\n")
    return documents

if __name__ == "__main__":
    docs = process_documents()
    if docs:
        print("\nExibindo metadados do primeiro lote contido:")
        print(docs[0].metadata)
