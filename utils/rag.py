import os
from pathlib import Path
import pandas as pd
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

# Cache the vectorstore to avoid reloading between queries
_vectordb_cache = None

def load_documents(folder_path: str = "data") -> List[Document]:
    """Loads all files in a folder and attaches parsed instructions."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Knowledge base not found: {folder_path}")

    splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    # --- Parse instructions.txt ---
    instructions_path = folder / "instructions.txt"
    if not instructions_path.exists():
        raise FileNotFoundError("Instructions file not found in the data folder.")

    instructions_map = {}
    current_file = None
    current_instructions = []
    column_map = {}
    tag_map = {}

    with open(instructions_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.endswith(":"):
                # Store the previous block if one exists
                if current_file:
                    instructions_map[current_file] = {
                        "instructions": "\n".join(current_instructions).strip(),
                        "columns": column_map.get(current_file, []),
                        "tag": tag_map.get(current_file),
                    }

                # Parse filename and tag from the new header line
                header = stripped[:-1]
                if "#" in header:
                    current_file, tag = header.split("#", 1)
                    current_file = current_file.strip()
                    tag_map[current_file] = tag.strip()
                else:
                    current_file = header.strip()
                    tag_map[current_file] = None

                # Reset for the new block
                current_instructions = []
                column_map[current_file] = []

            elif "-" in stripped:
                col, desc = stripped.split("-", 1)
                column_map[current_file].append((col.strip(), desc.strip()))
                current_instructions.append(stripped)
            else:
                current_instructions.append(stripped)

        # Save the last block
        if current_file:
            instructions_map[current_file] = {
                "instructions": "\n".join(current_instructions).strip(),
                "columns": column_map.get(current_file, []),
                "tag": tag_map.get(current_file),
            }

    # --- Load and chunk each file in folder ---
    docs = []
    for file in folder.glob("*.*"):
        print(f'Processing {file}')
        filename = file.name
        
        if "instructions" in file.name.lower():
            print(f'Skipped {file.name.lower()}')
            continue
            
        base_name = filename.split("#")[0].strip()

        # Match instructions block by base name
        info = instructions_map.get(filename) or instructions_map.get(base_name)
        if info is None:
            info = {"instructions": "No specific instructions provided.", "columns": [], "tag": None}

        instructions = info["instructions"]
        tag = info["tag"]
        columns_to_extract = [col for col, _ in info["columns"]]

        # --- Handle text / markdown files ---
        if file.suffix.lower() in [".txt", ".md"]:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                for chunk in splitter.split_text(text):
                    docs.append(Document(page_content=chunk, metadata={
                        "source": str(file),
                        "instructions": instructions,
                        "tag": tag
                    }))

        # --- Handle PDFs ---
        elif file.suffix.lower() == ".pdf":
            try:
                loader = PyPDFLoader(str(file))
                for d in loader.load():
                    for chunk in splitter.split_text(d.page_content):
                        docs.append(Document(page_content=chunk, metadata={
                            "source": str(file),
                            "instructions": instructions,
                            "tag": tag
                        }))
            except Exception as e:
                print(f"Warning: Could not read PDF {file}: {e}")

        # --- Handle Excel files ---
        elif file.suffix.lower() in [".xls", ".xlsx"]:
            try:
                excel_data = pd.ExcelFile(file)
                for sheet_name in excel_data.sheet_names:
                    # Read sheet as strings and don't drop empty columns
                    df = pd.read_excel(file, sheet_name=sheet_name, dtype=str, engine="openpyxl")
                    
                    # Normalize column names: strip whitespace and control chars
                    df.columns = [str(c).strip() for c in df.columns]

                    # Also normalize the columns_to_extract names (strip)
                    normalized_required = [c.strip() for c in columns_to_extract]

                    # Determine which required columns are available after normalization
                    available_cols = [c for c in normalized_required if c in df.columns]

                    if not available_cols:
                        print('No available columns')
                        continue
                    else:
                        print(f'Available cols: {available_cols}')

                    # Replace missing values with a placeholder so columns don't disappear for a row
                    df = df[available_cols].fillna("N/A")

                    text_lines = []
                    for _, row in df.iterrows():
                        # Preserve column order and always include the column name — show "N/A" when missing.
                        row_text = ", ".join(f"{col}: {row.get(col, 'N/A')}" for col in available_cols)
                        # skip completely empty rows (all N/A)
                        if all((str(row.get(col, "")).strip() in ("", "N/A") for col in available_cols)):
                            continue
                        text_lines.append(row_text)

                    # Only create docs if there is text
                    if text_lines:
                        for chunk in splitter.split_text("\n".join(text_lines)):
                            docs.append(Document(page_content=chunk, metadata={
                                "source": f"{file}:{sheet_name}",
                                "instructions": instructions,
                                "tag": tag
                            }))
            except Exception as e:
                print(f"Warning: Could not process Excel file {file}: {e}")

    return docs


def build_vectorstore(folder_path: str = "data"):
    """Builds and persists a Chroma vectorstore from the given data folder."""
    docs = load_documents(folder_path)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="knowledge_base",
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb


def search_knowledge_base(query: str, tag: str = None, folder_path: str = "data") -> str:
    """Search the knowledge base without rebuilding embeddings every query."""
    global _vectordb_cache
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    collection_name = "knowledge_base"

    if _vectordb_cache is None:
        if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
            print("Loading existing Chroma vectorstore...")
            _vectordb_cache = Chroma(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
            )
        else:
            print("No existing vectorstore found. Building a new one...")
            _vectordb_cache = build_vectorstore(folder_path)

    vectordb = _vectordb_cache
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    if tag:
        docs = [d for d in docs if (d.metadata.get("tag") or "").lower() == tag.lower()]

    if not docs:
        return "No relevant documents found."

    return "\n\n".join([
        f"Source: {d.metadata.get('source')}\n"
        f"Tag: {d.metadata.get('tag')}\n"
        f"Instructions: {d.metadata.get('instructions')}\n\n"
        f"{d.page_content}"
        for d in docs
    ])
