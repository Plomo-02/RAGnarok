import PyPDF2
import nest_asyncio
import os
nest_asyncio.apply()
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
load_dotenv()

class ExtractorManager:
    def __init__(self,input_path): 
        self.input_path = input_path

    def extract_text_from_pdf(self):
        for x, file_pdf in enumerate(os.listdir(self.input_path)):
            file_pdf = os.path.join(self.input_path, file_pdf)
            with open(file_pdf, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            with open(f'temp_{x}.txt', 'w', encoding='UTF-8') as file:
                file.write(text)


    def extract_text_llama(self):
        current_file = Path(__file__)

    # Trova la cartella "RAGnarok" risalendo nelle directory
        project_root = current_file.parents
        for folder in project_root:
            if folder.name == "RAGnarok":
                rag_path = folder
                break
        file_path = Path("txt_input") / f"llama_output{x}.txt"

    # Crea la directory genitore se non esiste
        file_path.parent.mkdir(parents=True, exist_ok=True)

        parser = LlamaParse(
            result_type="markdown",  # "markdown" and "text" are available
            verbose=True,
        )

        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(#da cambiare la directory di input 
            self.input_path, file_extractor=file_extractor
        ).load_data()
        for x in range(len(documents)):
            with open(f"txt_input/llama_output{x}.txt", "w", encoding = 'utf-8') as file:
                for document in documents:
                    file.write(document.text + "\n\n")


def get_input_path():

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

if __name__ == "__main__":
    input_path = get_input_path()
    extractor = ExtractorManager(input_path)
    #extractor.extract_text_from_pdf()
    extractor.extract_text_llama()




# from pathlib import Path
# import os # Potrebbe servire per altre parti, ma pathlib gestisce i percorsi
# from llama_parse import LlamaParse # Assumo sia importato correttamente
# from llama_index.core import SimpleDirectoryReader # Assumo sia importato correttamente

# class YourClassName: # Assumi che questo sia dentro una classe
#     def __init__(self, input_path="your_input_directory"): # Definisci input_path
#         self.input_path = input_path

#     def find_project_root(self, marker_name="RAGnarok") -> Path:
#         """
#         Cerca verso l'alto dalla directory dello script corrente
#         per trovare una directory con il nome specificato.
#         """
#         try:
#             current_path = Path(__file__).resolve()
#         except NameError:
#             current_path = Path.cwd().resolve()
#             print(f"Attenzione: __file__ non definito. Uso la directory corrente: {current_path}")

#         project_root = None
#         for parent in current_path.parents:
#             if parent.name == marker_name:
#                 project_root = parent
#                 break

#         if project_root is None:
#              if current_path.name == marker_name:
#                  project_root = current_path
#              else:
#                  raise FileNotFoundError(
#                      f"Impossibile trovare la directory radice del progetto '{marker_name}' "
#                      f"risalendo da {current_path}"
#                  )
#         return project_root

#     def extract_text_llama(self):
#         try:
#             # 1. Trova la radice del progetto "RAGnarok"
#             rag_path = self.find_project_root(marker_name="RAGnarok")
#             print(f"Trovata radice del progetto: {rag_path}")

#             # 2. Definisci la sottodirectory di output RELATIVA alla radice
#             output_subdir_name = "txt_input"
#             output_dir_path = rag_path / output_subdir_name

#             # 3. Assicura che la directory di output esista
#             output_dir_path.mkdir(parents=True, exist_ok=True)
#             print(f"Assicurata l'esistenza della directory di output: {output_dir_path}")

#             # --- Configurazione LlamaParse ---
#             parser = LlamaParse(
#                 result_type="markdown",  # "markdown" and "text" are available
#                 verbose=True,
#             )
#             file_extractor = {".pdf": parser}

#             # --- Lettura documenti ---
#             # Assicurati che self.input_path sia definito correttamente
#             # Potrebbe essere relativo a rag_path o un percorso assoluto
#             # Esempio: input_dir = rag_path / self.input_path se è relativo
#             input_dir = Path(self.input_path) # Assumi sia già il percorso corretto
#             print(f"Lettura documenti da: {input_dir}")
#             documents = SimpleDirectoryReader(
#                 input_dir=str(input_dir), # SimpleDirectoryReader vuole una stringa
#                 file_extractor=file_extractor
#             ).load_data()
#             print(f"Caricati {len(documents)} documenti.")

#             # --- Scrittura output ---
#             # Cicla sui documenti CARICATI per creare un file per ciascuno?
#             # O vuoi scrivere TUTTI i documenti in UN file?
#             # Il codice originale sembrava voler scrivere tutti i doc in ogni file, il che è strano.
#             # Modifico per creare UN file per OGNI documento caricato.

#             if not documents:
#                  print("Nessun documento caricato, nessuna scrittura eseguita.")
#                  return

#             for idx, document in enumerate(documents):
#                 # Costruisci il percorso del file di output per QUESTO documento
#                 # idx è l'indice (0, 1, 2, ...)
#                 output_filename = f"llama_output{idx}.txt"
#                 file_path = output_dir_path / output_filename

#                 print(f"Scrittura del documento {idx+1}/{len(documents)} su: {file_path}")
#                 try:
#                     # Scrivi il testo del singolo documento nel file corrispondente
#                     with open(file_path, "w", encoding='utf-8') as file:
#                         if hasattr(document, 'text') and document.text:
#                              file.write(str(document.text)) # Scrivi solo il testo del doc corrente
#                              # file.write("\n\n") # Aggiungi newline se vuoi separare qualcosa dopo?
#                         else:
#                              print(f"  Attenzione: Documento {idx} non ha attributo 'text' o è vuoto.")
#                 except IOError as e:
#                      print(f"Errore durante la scrittura del file {file_path}: {e}")
#                 except Exception as e:
#                      print(f"Errore inatteso durante la scrittura del file {file_path}: {e}")

#             print("Processo di estrazione e scrittura completato.")

#         except FileNotFoundError as e:
#             print(f"Errore critico (radice progetto o input non trovato?): {e}")
#         except Exception as e:
#             print(f"Errore generale in extract_text_llama: {e}")
#             # Considera di loggare l'intero traceback per debug
#             import traceback
#             traceback.print_exc()

# # Esempio di come potresti usarlo:
# # Assicurati che esista una directory 'input_pdfs' dentro 'RAGnarok' o dove ti aspetti
# # if __name__ == "__main__":
# #    extractor_instance = YourClassName(input_path="percorso/alla/tua/cartella/pdf") # Specifica il percorso input
# #    extractor_instance.extract_text_llama()