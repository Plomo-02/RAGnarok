import PyPDF2
import nest_asyncio
import os
nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
load_dotenv()

class ExtractorManager:
    def __init__(self,input_path): 
        self.input_path = input_path

    def extract_text_from_pdf(self):
        for x,file_pdf in enumerate(os.listdir(self.input_path)):
            with open(file_pdf, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            with open(f'temp_{x}.txt', 'w') as file:
                file.write(text)


    def extract_text_llama(input_path):

        parser = LlamaParse(
            result_type="markdown",  # "markdown" and "text" are available
            verbose=True,
        )

        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(#da cambiare la directory di input 
            input_path, file_extractor=file_extractor
        ).load_data()
        for x in range(len(documents)):
            with open(f"txt_input/llama_output{x}.txt", "w") as file:
                for document in documents:
                    file.write(document.text + "\n\n")


def get_input_path():
    from pathlib import Path

    # Ottieni il percorso del file corrente (read.py)
    current_file = Path(__file__)

    # Trova la cartella "RAGnarok" risalendo nelle directory
    project_root = current_file.parents
    for folder in project_root:
        if folder.name == "RAGnarok":
            rag_path = folder
            break
    else:
        raise FileNotFoundError("Cartella 'RAGnarok' non trovata!")

    # Percorso della cartella frontend/temp
    files_temp_path = rag_path / "frontend" / "temp"
    return files_temp_path


   




