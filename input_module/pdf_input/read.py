import PyPDF2
import nest_asyncio
import os
import logging
nest_asyncio.apply()
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from input_module.utils.tools import find_project_root
from dotenv import load_dotenv
load_dotenv()

class ExtractorManager:
    def __init__(self,input_path : str): 
        self.input_path = Path(input_path)
        self._create_output_dir()


    def _create_output_dir(self):
        
        rag_path = find_project_root(marker_name="RAGnarok")
        output_subdir_name = "txt_input"
        self.output_dir_path = rag_path / output_subdir_name
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

    def _extract_text_from_pdf(self):


        try:
            # 1. Trova la radice del progetto (assumendo che la funzione esista)
            # Se find_project_root non è disponibile, dovrai definirla o usare un percorso fisso/relativo
            rag_path = find_project_root(marker_name="RAGnarok")
            if rag_path is None:
                    logging.error("Impossibile trovare la directory radice del progetto 'RAGnarok'.")
                    return # Esce se non trova la root

        except NameError: # Gestisce il caso in cui find_project_root non sia definito
             logging.error("La funzione 'find_project_root' non è definita.")
             return # Esce se non può determinare la root

        # 2. Definisci e crea la directory di output

        try:
            self.output_dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Directory di output assicurata: {self.output_dir_path}")
        except OSError as e:
            logging.error(f"Impossibile creare la directory di output {self.output_dir_path}: {e}")
            return # Esce se non può creare la dir di output

        logging.info(f"Inizio estrazione testo da PDF in: {self.input_path}")

        # 3. Itera sui file nella directory di input usando pathlib
        file_count = 0
        processed_count = 0
        for input_file_path in self.input_path.iterdir():
            # 4. Filtra: processa solo file con estensione .pdf (case-insensitive)
            if input_file_path.is_file() and input_file_path.suffix.lower() == '.pdf':
                file_count += 1
                # 5. Crea il nome del file di output basato sull'originale
                # Usa .stem per ottenere il nome file senza estensione
                base_filename = input_file_path.stem
                # Crea il nuovo nome file con estensione .txt
                output_filename = f"{base_filename}.txt"
                # Crea il percorso completo per il file di output
                output_file_path = self.output_dir_path / output_filename

                logging.info(f"Processando: {input_file_path.name} -> {output_file_path}")

                try:
                    # 6. Estrai testo dal PDF
                    extracted_text = ''
                    with open(input_file_path, 'rb') as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        # Aggiungi controllo per PDF criptati (opzionale)
                        # if reader.is_encrypted:
                        #     logging.warning(f"Il file {input_file_path.name} è criptato e non può essere processato.")
                        #     continue # Salta al prossimo file

                        for page_num, page in enumerate(reader.pages):
                            page_text = page.extract_text()
                            if page_text: # Aggiungi solo se l'estrazione ha prodotto testo
                                extracted_text += page_text + "\n" # Aggiungi a capo tra le pagine (opzionale)
                            # else:
                            #     logging.debug(f"Nessun testo estratto da pagina {page_num + 1} di {input_file_path.name}")


                    # 7. Scrivi il testo estratto nel file di output
                    if extracted_text: # Scrivi solo se è stato estratto del testo
                        with open(output_file_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(extracted_text)
                        processed_count += 1
                        logging.info(f"Testo estratto e salvato in: {output_file_path}")
                    else:
                        logging.warning(f"Nessun testo estratto da {input_file_path.name}. Nessun file .txt creato.")

                except PyPDF2.errors.PdfReadError as e:
                    logging.error(f"Errore durante la lettura di {input_file_path.name}: {e}. File saltato.")
                except Exception as e: # Cattura altri errori imprevisti per un singolo file
                    logging.error(f"Errore imprevisto durante l'elaborazione di {input_file_path.name}: {e}. File saltato.")

            else: # Opzionale: logga i file ignorati
                if input_file_path.is_file():
                    logging.debug(f"Ignorato file non PDF: {input_file_path.name}")
                elif input_file_path.is_dir():
                    logging.debug(f"Ignorata sottodirectory: {input_file_path.name}")

        logging.info(f"Processo completato. Trovati {file_count} file PDF. Estratto testo da {processed_count} file.")

    def _is_extracted_text_valid(self, text, min_words_per_page=10, min_valid_chars_ratio=0.7):
        if not text.strip():
            return False
        
        # Controllo della lunghezza minima
        words = text.split()
        if len(words) < min_words_per_page:
            return False
        
        # Controllo del rapporto di caratteri validi
        valid_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,;:!?()-')
        total_chars = len(text)
        
        if total_chars == 0 or valid_chars / total_chars < min_valid_chars_ratio:
            return False
        
        # Controllo di pattern anomali (caratteri ripetuti, etc.)
        unusual_patterns = ['���', '###', '...', '   ']
        for pattern in unusual_patterns:
            if pattern in text:
                return False
        
        return True

    def _extract_text_llama(self):
        try:
            # 1. Trova la radice del progetto "RAGnarok"
            rag_path = find_project_root(marker_name="RAGnarok")

            # 2. Definisci la sottodirectory di output RELATIVA alla radice
            output_subdir_name = "txt_input"
            output_dir_path = rag_path / output_subdir_name

            # 3. Assicura che la directory di output esista
            output_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Assicurata l'esistenza della directory di output: {output_dir_path}")

            # --- Configurazione LlamaParse ---
            parser = LlamaParse(
                result_type="markdown",  # "markdown" and "text" are available
                verbose=True,
            )
            file_extractor = {".pdf": parser}

            # --- Lettura documenti ---
            # Assicurati che self.input_path sia definito correttamente
            # Potrebbe essere relativo a rag_path o un percorso assoluto
            # Esempio: input_dir = rag_path / self.input_path se è relativo
            input_dir = Path(self.input_path) # Assumi sia già il percorso corretto
            print(f"Lettura documenti da: {input_dir}")
            documents = SimpleDirectoryReader(
                input_dir=str(input_dir), # SimpleDirectoryReader vuole una stringa
                file_extractor=file_extractor
            ).load_data()
            print(f"Caricati {len(documents)} documenti.")

            # --- Scrittura output ---
            # Cicla sui documenti CARICATI per creare un file per ciascuno?
            # O vuoi scrivere TUTTI i documenti in UN file?
            # Il codice originale sembrava voler scrivere tutti i doc in ogni file, il che è strano.
            # Modifico per creare UN file per OGNI documento caricato.

            if not documents:
                 print("Nessun documento caricato, nessuna scrittura eseguita.")
                 return

            for idx, document in enumerate(documents):
                # Costruisci il percorso del file di output per QUESTO documento
                # idx è l'indice (0, 1, 2, ...)
                output_filename = f"llama_output{idx}.txt"
                file_path = output_dir_path / output_filename

                print(f"Scrittura del documento {idx+1}/{len(documents)} su: {file_path}")
                try:
                    # Scrivi il testo del singolo documento nel file corrispondente
                    with open(file_path, "w", encoding='utf-8') as file:
                        if hasattr(document, 'text') and document.text:
                             file.write(str(document.text)) # Scrivi solo il testo del doc corrente
                             # file.write("\n\n") # Aggiungi newline se vuoi separare qualcosa dopo?
                        else:
                             print(f"  Attenzione: Documento {idx} non ha attributo 'text' o è vuoto.")
                except IOError as e:
                     print(f"Errore durante la scrittura del file {file_path}: {e}")
                except Exception as e:
                     print(f"Errore inatteso durante la scrittura del file {file_path}: {e}")

            print("Processo di estrazione e scrittura completato.")

        except FileNotFoundError as e:
            print(f"Errore critico (radice progetto o input non trovato?): {e}")
        except Exception as e:
            print(f"Errore generale in extract_text_llama: {e}")
            # Considera di loggare l'intero traceback per debug
            import traceback
            traceback.print_exc()

    def _extract_ocr(self, file_pdf):
        pass

    def _validate_ocr_extraction(self, file_pdf):
        pass

    def _decide_extraction (self):
        for file_pdf in os.listdir(self.output_dir_path):
            with open(file_pdf, 'r') as file:
                content = file.read()
            if self._is_extracted_text_valid(content) is False:
                self._extract_ocr(file_pdf)
                result = self._validate_ocr_extraction(file_pdf)
                if result is False:
                    self._extract_text_llama()
                               

    def extract_text(self):
        
        type_of_extr =  self._decide_extraction()
        if type_of_extr == "pdf":
            self._extract_text_from_pdf()
        elif type_of_extr == "llama":
            self._extract_text_llama()
        elif type_of_extr == "ocr":
            self._extract_ocr()
        else:
            print("Tipo di estrazione non supportato.")

