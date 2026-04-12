from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from config.settings import CHROMA_DB_PATH, PROJECT_ROOT, embeddings


def load_pdfs(docs_dir: Path) -> list[Document]:
    """
    Load all PDF files from a directory into LangChain Documents.

    Args:
        docs_dir: Directory containing PDF files.

    Returns:
        A list of loaded documents.
    """
    documents: list[Document] = []

    if not docs_dir.exists() or not docs_dir.is_dir():
        print(f"Documents directory does not exist: {docs_dir}")
        return documents

    pdf_files = sorted(docs_dir.glob("*.pdf"))

    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        loaded_docs = loader.load()
        documents.extend(loaded_docs)

    return documents


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks for vector ingestion.

    Args:
        docs: Source documents.

    Returns:
        A list of chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(docs)


def _create_sample_documents() -> list[Document]:
    """
    Create sample fallback documents with Chinese car export knowledge.

    Returns:
        A list containing one or more LangChain Documents.
    """
    sample_text = """
Chinese car export overview:

BYD is one of China's leading new energy vehicle manufacturers. Popular export models include:
- BYD Dolphin: compact EV hatchback suitable for urban driving
- BYD Atto 3: compact electric SUV popular in many overseas markets
- BYD Seal: electric sedan positioned for mid-size buyers
- BYD Han: premium sedan with strong EV and PHEV positioning
BYD vehicles are often exported with FOB China or CIF destination port pricing in USD.

Chery is a major Chinese automotive exporter known for affordability and broad market coverage. Popular models include:
- Chery Tiggo 2: entry-level compact SUV
- Chery Tiggo 4: practical compact crossover
- Chery Tiggo 7: mid-size SUV with export popularity
- Chery Tiggo 8: 7-seat SUV for family and commercial buyers
Chery is active in Latin America, the Middle East, Africa, and Eastern Europe.

MG is a globally recognized brand under SAIC Motor. Popular export models include:
- MG ZS: compact SUV available in petrol and EV versions
- MG5: sedan/wagon platform popular for value-focused buyers
- MG4 EV: modern electric hatchback with growing international demand
- MG HS: larger SUV for family use
MG often appeals to buyers seeking a balance of price, technology, and international brand familiarity.

Geely is one of China's largest automotive groups with strong engineering capability. Popular export models include:
- Geely Coolray: compact SUV known for competitive pricing
- Geely Emgrand: sedan offering practical value
- Geely Azkarra: crossover SUV in selected export markets
- Geely Geometry series: electric-focused models in some regions
Geely benefits from global partnerships and improved product quality perception.

SAIC Motor is one of China's biggest automotive companies and the parent group behind MG and other brands. In export business,
SAIC products are known for large-scale production, broad parts support, and competitive pricing structures.
For international trade, quotations are commonly prepared in USD under FOB Shanghai, FOB Ningbo, or CIF destination terms.

General export notes:
- FOB means Free On Board, where the seller delivers goods onto the vessel at the Chinese port of shipment.
- CIF means Cost, Insurance, and Freight, where the seller also covers ocean freight and insurance to the destination port.
- Chinese car exporters commonly prepare offers with model year, trim level, battery or engine specs, MOQ, lead time, and Incoterms.
- Buyers often compare EVs and SUVs from BYD, Chery, MG, Geely, and SAIC for fleet, dealership, and retail import programs.
""".strip()

    return [
        Document(
            page_content=sample_text,
            metadata={
                "source": "sample_knowledge_base",
                "type": "fallback",
                "domain": "chinese_car_export",
            },
        )
    ]


def ingest_documents(docs_dir: Optional[Path] = None) -> int:
    """
    Load, split, and ingest documents into a Chroma vector store.

    If no PDFs are found, fallback sample knowledge documents are used.

    Args:
        docs_dir: Optional directory containing PDF files.

    Returns:
        Number of chunks ingested.
    """
    target_docs_dir = docs_dir if docs_dir is not None else PROJECT_ROOT / "data" / "docs"
    target_docs_dir = Path(target_docs_dir)

    raw_docs = load_pdfs(target_docs_dir)

    if not raw_docs:
        print("No PDF documents found. Creating sample car export knowledge document.")
        raw_docs = _create_sample_documents()

    chunks = split_documents(raw_docs)

    chroma_path = Path(CHROMA_DB_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_path),
    )
    # Note: Chroma >= 0.4 auto-persists, no need to call .persist()

    return len(chunks)


if __name__ == "__main__":
    count = ingest_documents()
    print(f"Ingested {count} document chunks")