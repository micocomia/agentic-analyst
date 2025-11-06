import os
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

def load_documents(folder_path: str = "data") -> list[Document]:
    docs = []
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Knowledge base not found: {folder_path}")
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    ### Load instructions from a text file
    instructions_path = folder / "instructions.txt"
    if not instructions_path.exists():
        raise FileNotFoundError("Instructions file not found.")
    
    ### Parse multi-line instructions
    instructions_map = {}
    column_map = {}
    current_file = None
    current_instructions = []
    with open(instructions_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.endswith(":"):  # New file block
                if current_file:
                    instructions_map[current_file] = {
                        'instructions': "\n".join(current_instructions),
                        'columns': column_map.get(current_file, [])
                    }
                current_file = stripped_line[:-1]  # Filename without colon
                current_instructions = []
                column_map[current_file] = []  # Reset column map for this file
            elif '-' in stripped_line: # Parsing column-specific descriptions
                col_name, col_instruction = stripped_line.split('-', 1)
                column_map[current_file].append((col_name.strip(), col_instruction.strip()))
            else:
                current_instructions.append(stripped_line)
    
    # Ensure last file instructions and columns are stored
    if current_file:
        instructions_map[current_file] = {
            'instructions': "\n".join(current_instructions),
            'columns': column_map.get(current_file, [])
        }

    # Chunk file data so that they can be converted into embeddings
    for file in folder.glob("*.*"):
        filename = file.name
        instructions_data = instructions_map.get(filename, {"instructions": "No specific instructions provided.", "columns": []})
        instructions = instructions_data['instructions']
        columns_to_extract = [col for col, _ in instructions_data['columns']]

        if file.suffix.lower() in [".txt", ".md"]:  # Handle text files
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = splitter.split_text(text)
                docs.extend([Document(page_content=chunk, metadata={"source": str(file), "instructions": instructions}) for chunk in chunks])
        
        elif file.suffix.lower() == ".pdf":  # Handle PDF files
            loader = PyPDFLoader(str(file))
            pdf_docs = loader.load()
            for d in pdf_docs:
                chunks = splitter.split_text(d.page_content)
                docs.extend([Document(page_content=chunk, metadata={"source": str(file), "instructions": instructions}) for chunk in chunks])

        elif file.suffix.lower() in [".xls", ".xlsx"]:  # Handle Excel files
            excel_data = pd.ExcelFile(file)
            for sheet_name in excel_data.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet_name, usecols=columns_to_extract)

                text_lines = []
                for _, row in df.iterrows():
                    row_text = ", ".join(f"{col}: {row[col]}" for col in columns_to_extract if col in df.columns)
                    text_lines.append(row_text)
                
                text = "\n".join(text_lines)
                chunks = splitter.split_text(text)
                docs.extend([Document(page_content=chunk, metadata={"source": f"{str(file)}:{sheet_name}", "instructions": instructions}) for chunk in chunks])

    return docs

def build_vectorstore(folder_path: str = "data"):
    docs = load_documents(folder_path)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="knowledge_base",
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    )
    return vectordb

def search_knowledge_base(query: str, folder_path: str = "data") -> str:
    vectordb = build_vectorstore(folder_path)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([f"Source: {d.metadata.get('source')}\nInstructions: {d.metadata.get('instructions')}\n{d.page_content}" for d in docs])