from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from RAGnarok.input_module.pdf_input.read import get_input_path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
import tiktoken # Importato per mostrare il conteggio dei token (opzionale per lo split)
import shutil
import nltk
import re 
import numpy as np
from langchain.schema.document import Document


class ChunkManager:
    def __init__(self, chunk_size = 1000, chunk_overlap  = 0.2, breakpoint_threshold_type = "percentile"):
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _semantic_chunking(self):
        # This is a long document we can split up.
        directory_input = get_input_path()
        for index,file in enumerate(os.listdir(directory_input)):
            with open(file) as f:
                state_of_the_union = f.read()
            text_splitter = SemanticChunker(OpenAIEmbeddings(), self.breakpoint_threshold_type)
            docs = text_splitter.create_documents([state_of_the_union])
            with open(f"chunked_{file}_{index}.txt", "w") as f:
                for doc in docs:
                    f.write(doc[0].page_content + "\n\n")

    def _recursive_chunking(self):
        directory_input = get_input_path()
        for index,file in enumerate(os.listdir(directory_input)):
            loader = PyPDFLoader(file)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            naive_chunks = text_splitter.split_documents(documents)
            for chunk in naive_chunks[10:15]:
                print(chunk.page_content+ "\n")
                with open(f"chunked_texts/chunked_{file}_{index}.txt", "w") as f:
                    f.write(chunk.page_content + "\n\n")

    def _fixed_size_chunking (self, chunk_size, chunk_overlap, encoding_name="cl100k_base"):
        directory_input = get_input_path()
        for index,file in enumerate(os.listdir(directory_input)):
            loader = PyPDFLoader(file)
            documents = loader.load()
            text_splitter = TokenTextSplitter(
                encoding_name=encoding_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            return text_splitter.split_documents(documents)

    def _analyze_document_content(self, text: str) -> dict:
        """Analizza il contenuto testuale per estrarre metriche utili."""
        analysis = {
            "char_count": 0,
            "line_count": 0,
            "paragraph_count": 0,
            "sentence_count": 0,
            "avg_line_length": 0,
            "avg_paragraph_length": 0, # In caratteri
            "avg_sentence_length": 0, # In caratteri
            "sentence_length_std": 0, # Deviazione standard lunghezza frasi
            "nltk_success": False,
            "is_likely_prose": False,
            "has_strong_paragraph_structure": False,
        }
        if not text or not isinstance(text, str) or len(text) < 50: # Ignora testi troppo corti
            return analysis

        analysis["char_count"] = len(text)

        # Analisi Strutturale Base
        lines = text.split('\n')
        analysis["line_count"] = len(lines)
        analysis["avg_line_length"] = analysis["char_count"] / analysis["line_count"] if analysis["line_count"] > 0 else 0
        # Conta paragrafi come blocchi separati da almeno una linea vuota
        paragraphs = re.split(r'\n\s*\n', text.strip()) # Split su una o più linee vuote
        analysis["paragraph_count"] = len(paragraphs)
        analysis["avg_paragraph_length"] = analysis["char_count"] / analysis["paragraph_count"] if analysis["paragraph_count"] > 0 else 0

        # Heuristica per struttura a paragrafi forte (Threshold da affinare!)
        paragraph_density = analysis["paragraph_count"] / analysis["line_count"] if analysis["line_count"] > 0 else 0
        if paragraph_density > 0.04 and analysis["paragraph_count"] > 3 and analysis["avg_paragraph_length"] < 2000:
             analysis["has_strong_paragraph_structure"] = True

        # Analisi Linguistica (NLTK)
        try:
            sentences = nltk.sent_tokenize(text)
            analysis["sentence_count"] = len(sentences)
            if analysis["sentence_count"] > 1: # Ha senso calcolare solo se ci sono più frasi
                sentence_lengths = [len(s) for s in sentences]
                analysis["avg_sentence_length"] = np.mean(sentence_lengths)
                analysis["sentence_length_std"] = np.std(sentence_lengths)
                analysis["nltk_success"] = True

                # Heuristica per prosa (Threshold da affinare!)
                # Lunghezza media ragionevole e deviazione standard non ECCESSIVAMENTE alta
                if 50 < analysis["avg_sentence_length"] < 1200 and analysis["sentence_length_std"] < analysis["avg_sentence_length"] * 1.5:
                    analysis["is_likely_prose"] = True

        except Exception as e:
            print(f"Analysis: NLTK sentence tokenization failed during analysis: {e}.")
            analysis["nltk_success"] = False

        return analysis

        
    def _recommend_chunking_strategy(self, analysis: dict) -> str:
        """Sceglie la strategia basandosi sui risultati dell'analisi."""

        # 1. Priorità alla struttura a paragrafi se forte
        if analysis["has_strong_paragraph_structure"]:
            print("Analysis Recommendation: Strong paragraph structure detected -> 'section'")
            return "section"

        # 2. Se è prosa, considera Semantico o Sentence
        if analysis["is_likely_prose"]:
            # Preferisci semantico se richiesto e possibile?
            if self.prefer_semantic_in_auto and self.openai_api_key_present:
                    print("Analysis Recommendation: Prose detected, semantic preferred and possible -> 'semantic'")
                    return "semantic"
            else:
                    # Altrimenti usa sentence (NLTK deve aver funzionato per is_likely_prose=True)
                    print("Analysis Recommendation: Prose detected -> 'sentence'")
                    return "sentence"

        # 3. Fallback generico se non è prosa chiaramente strutturata o NLTK ha fallito
        # Potresti aggiungere qui logica per rilevare codice/liste e usare recursive
        print("Analysis Recommendation: No clear prose or paragraph structure detected -> 'recursive' (fallback)")
        return "recursive"


    def _automatic_chunking(self, documents: list[Document]) -> list[Document]:
        """
        Analizza il contenuto e applica automaticamente la strategia di chunking ritenuta migliore.
        """
        # Analizza un campione rappresentativo (es. i primi N documenti o i primi X caratteri totali)
        representative_text = ""
        chars_to_analyze = 5000 # Analizza fino a X caratteri
        for doc in documents:
            if doc.page_content and isinstance(doc.page_content, str):
                representative_text += doc.page_content + "\n\n" # Aggiungi separatore tra pagine/doc
                if len(representative_text) >= chars_to_analyze:
                    break
        representative_text = representative_text[:chars_to_analyze]

        if not representative_text:
             print("Automatic Chunking: No text content found to analyze. Falling back to recursive.")
             return self._recursive_chunking(documents)

        # Esegui l'analisi
        analysis_results = self._analyze_document_content(representative_text)
        # Stampa riassunto analisi (utile per debug)
        print("\n--- Document Analysis Summary ---")
        for key, value in analysis_results.items():
             if isinstance(value, float):
                 print(f"  {key}: {value:.2f}")
             else:
                 print(f"  {key}: {value}")
        print("-----------------------------")


        # Ottieni la strategia raccomandata
        chosen_strategy = self._recommend_chunking_strategy(analysis_results)

        print(f"\nAutomatic Chunking: Applying '{chosen_strategy}' strategy based on analysis...")

        # Applica la strategia scelta
        if chosen_strategy == "section":
            # Usa il separatore di paragrafo standard
            return self._section_chunking_by_separator(documents, separator="\n\n")
        elif chosen_strategy == "sentence":
             return self._sentence_aware_chunking(documents) # Già contiene fallback a recursive se NLTK fallisce
        elif chosen_strategy == "semantic":
             # Doppio controllo API Key qui è prudente
             if not self.openai_api_key_present:
                 print("Automatic Chunking Warning: Semantic recommended but API key missing. Falling back to sentence chunking.")
                 return self._sentence_aware_chunking(documents) # Fallback a sentence (che a sua volta può fare fallback a recursive)
             else:
                 return self._semantic_chunking(documents)
        else: # chosen_strategy == "recursive" o fallback inaspettato
            return self._recursive_chunking(documents)


   
    def _sentence_aware_chunking(
        text: str,
        max_tokens_per_chunk: int = 256,
        overlap_sentences: int = 1,
        encoding_name: str = "cl100k_base",
        verbose: bool = False
    ) -> list[str]:
        """
        Divide il testo in chunk rispettando i confini delle frasi.

        Args:
            text: Il testo di input da dividere.
            max_tokens_per_chunk: Il numero massimo desiderato di token per chunk.
                                Il chunk finale potrebbe superarlo leggermente a causa
                                dell'aggiunta dell'ultima frase.
            overlap_sentences: Il numero di frasi da sovrapporre tra chunk consecutivi.
            encoding_name: Il nome della codifica tiktoken da usare per contare i token.
            verbose: Se True, stampa informazioni dettagliate durante il processo.

        Returns:
            Una lista di stringhe, dove ogni stringa è un chunk di testo.
        """
        if not isinstance(text, str) or not text.strip():
            return []
        if max_tokens_per_chunk <= 0:
            raise ValueError("max_tokens_per_chunk deve essere positivo.")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences non può essere negativo.")

        tokenizer = tiktoken.get_encoding(encoding_name)

        # 1. Dividi in frasi
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        if verbose:
            print(f"Testo diviso in {len(sentences)} frasi.")

        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        sentence_index = 0

        while sentence_index < len(sentences):
            sentence = sentences[sentence_index]
            # Usa l'encoding corretto per contare i token della frase corrente
            sentence_tokens = len(tokenizer.encode(sentence))

            # --- Gestione frase troppo lunga ---
            if sentence_tokens > max_tokens_per_chunk:
                if verbose:
                    print(f"Attenzione: Frase {sentence_index} è più lunga ({sentence_tokens} tokens) "
                        f"di max_tokens_per_chunk ({max_tokens_per_chunk}). Verrà aggiunta come chunk separato.")

                # Se c'è un chunk in costruzione, finalizzalo prima
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(chunk_text)
                    if verbose:
                        print(f"  Chunk finalizzato (terminante alla frase {sentence_index-1}) con {current_chunk_tokens} tokens.")
                    current_chunk_sentences = [] # Resetta per il prossimo giro
                    current_chunk_tokens = 0

                # Aggiungi la frase lunga come chunk a sé stante
                chunks.append(sentence)
                if verbose:
                    print(f"  Aggiunta frase lunga {sentence_index} come chunk separato.")

                sentence_index += 1 # Passa alla frase successiva
                continue # Salta il resto del ciclo per questa iterazione

            # --- Calcolo token potenziale se si aggiunge la frase corrente ---
            # Unisci le frasi temporaneamente per un conteggio più accurato (include spazi)
            potential_chunk_list = current_chunk_sentences + [sentence]
            potential_chunk_text = " ".join(potential_chunk_list)
            potential_tokens = len(tokenizer.encode(potential_chunk_text))

            # --- Verifica se aggiungere la frase supera il limite ---
            if potential_tokens <= max_tokens_per_chunk:
                # Aggiungi la frase al chunk corrente
                current_chunk_sentences.append(sentence)
                current_chunk_tokens = potential_tokens # Aggiorna con il conteggio accurato
                if verbose:
                    print(f"  Frase {sentence_index}: Aggiunta al chunk corrente. Tokens totali: {current_chunk_tokens}/{max_tokens_per_chunk}")
                sentence_index += 1 # Passa alla prossima frase da valutare
            else:
                # Limite superato: finalizza il chunk corrente (se non è vuoto)
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(chunk_text)
                    if verbose:
                        print(f"Chunk finalizzato (terminante alla frase {sentence_index-1}) con {current_chunk_tokens} tokens.")

                # --- Inizia un nuovo chunk con sovrapposizione ---
                # Quante frasi indietro iniziare per il nuovo chunk?
                # `sentence_index` è l'indice della frase che *non* ci stava.
                # Il chunk precedente terminava a `sentence_index - 1`.
                # Il nuovo chunk deve iniziare `overlap_sentences` prima della fine del precedente.
                # Indice di inizio per il *nuovo* chunk = max(0, indice_fine_precedente - overlap_sentences + 1)
                #                                     = max(0, (sentence_index - 1) - overlap_sentences + 1)
                #                                     = max(0, sentence_index - overlap_sentences)

                # Tuttavia, è più semplice pensare a quale indice RITORNARE per iniziare il prossimo chunk.
                # Se overlap=1, vogliamo che la *prossima* iterazione inizi dalla *ultima* frase del chunk appena creato.
                # Se overlap=0, la prossima iterazione inizia da `sentence_index`.
                # Se overlap=N, la prossima iterazione inizia da `max(0, sentence_index - N)`.

                # Calcola l'indice da cui la *prossima* iterazione del `while` dovrà ripartire
                start_next_iteration_index = max(0, sentence_index - overlap_sentences)

                # Siccome il loop incrementa `sentence_index` alla fine (o all'inizio del giro dopo `continue`),
                # dobbiamo impostare `sentence_index` qui in modo che al prossimo giro sia corretto.
                # Se l'indice di partenza calcolato è lo stesso di `sentence_index`, significa che
                # l'overlap non ha effetto (magari perché la frase era troppo lunga o siamo all'inizio).
                # Evitiamo loop infiniti.
                if start_next_iteration_index >= sentence_index and overlap_sentences > 0:
                    if verbose:
                        print(f"  Attenzione: Overlap ({overlap_sentences}) non applicabile o causerebbe stallo all'indice {sentence_index}. Procedo senza overlap forzato per questo passaggio.")
                    # In questo caso raro, si procede come se overlap fosse 0 per questo specifico passaggio
                    start_next_iteration_index = sentence_index # Forziamo l'avanzamento

                sentence_index = start_next_iteration_index

                if verbose:
                    print(f"  Inizio nuovo chunk. Prossima iterazione partirà dalla frase indice: {sentence_index}")

                # Resetta per il nuovo chunk
                current_chunk_sentences = []
                current_chunk_tokens = 0
                # La frase `sentence_index` (che prima non ci stava) verrà riconsiderata
                # all'inizio della prossima iterazione del while loop.


        # Aggiungi l'ultimo chunk rimasto, se ce n'è uno
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)
            if verbose:
                print(f"Chunk finale aggiunto con {current_chunk_tokens} tokens.")

        return chunks


    def _section_chunking_by_separator(
        text: str,
        separator: str = "\n\n",
        min_chunk_size: int = 1  # Opzionale: ignora chunk molto piccoli
    ) -> list[str]:
        """
        Divide il testo in chunk basandosi su un separatore specificato.

        Utile per dividere per paragrafi (default) o altri delimitatori.

        Args:
            text: Il testo di input da dividere.
            separator: La stringa usata come delimitatore per le sezioni.
                    Default: "\\n\\n" (doppio a capo, comune per i paragrafi).
            min_chunk_size: La dimensione minima (in caratteri) che un chunk
                            deve avere per essere incluso. Utile per scartare
                            separatori multipli o righe vuote/brevi.

        Returns:
            Una lista di stringhe, dove ogni stringa è un chunk (sezione).
        """
        if not isinstance(text, str) or not text.strip():
            return []
        if not isinstance(separator, str) or not separator:
            # Se il separatore non è valido, ritorna il testo intero come unico chunk
            # o solleva un errore, a seconda di come vuoi gestire il caso.
            # Qui lo trattiamo come un unico chunk se il testo non è vuoto.
            return [text.strip()] if text.strip() else []

        # Dividi il testo usando il separatore
        # Usiamo re.split per gestire meglio i separatori che potrebbero essere
        # espressioni regolari e per potenzialmente mantenere i separatori se necessario
        # (anche se qui li stiamo rimuovendo, che è il comportamento di default di split).
        # Usare string.split(separator) è spesso sufficiente per separatori semplici.
        # chunks = text.split(separator)
        chunks = re.split(f"({re.escape(separator)})", text) # re.split mantiene il separatore se catturato con ()
                                                            # ma crea più elementi nella lista, quindi dobbiamo gestirlo

        # Ricostruiamo i chunk, rimuovendo gli spazi e filtrando quelli troppo piccoli
        processed_chunks = []
        current_chunk = ""
        for i, part in enumerate(chunks):
            if i % 2 == 0: # Parti di testo tra i separatori
                current_chunk += part
            else: # Separatore stesso (catturato da re.split)
                # Decidi se vuoi includere il separatore o no.
                # Per la divisione per paragrafo, di solito NON lo includi.
                # current_chunk += part # -> Includerebbe il separatore alla fine del chunk precedente

                # Finalizza il chunk corrente se non è vuoto e rispetta la dimensione minima
                trimmed_chunk = current_chunk.strip()
                if trimmed_chunk and len(trimmed_chunk) >= min_chunk_size:
                    processed_chunks.append(trimmed_chunk)
                current_chunk = "" # Inizia un nuovo chunk virtuale dopo il separatore

        # Aggiungi l'ultimo pezzo di testo dopo l'ultimo separatore (o se non c'erano separatori)
        trimmed_last_chunk = current_chunk.strip()
        if trimmed_last_chunk and len(trimmed_last_chunk) >= min_chunk_size:
            processed_chunks.append(trimmed_last_chunk)

        # Alternativa più semplice se non ti serve mantenere i separatori e re.split confonde:
        # chunks_simple = text.split(separator)
        # processed_chunks = [chunk.strip() for chunk in chunks_simple if chunk.strip() and len(chunk.strip()) >= min_chunk_size]

        # Scegliamo l'implementazione più semplice con string.split per minimalismo
        chunks_simple = text.split(separator)
        final_chunks = [
            chunk.strip()
            for chunk in chunks_simple
            if chunk.strip() and len(chunk.strip()) >= min_chunk_size
        ]

        return final_chunks

    def _count_tokens(self, text):
        """Helper to count tokens."""
        return len(self.tokenizer.encode(text))

    def chunk(self, type = "semantic", include_metadata = False):
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix="chunked_texts_")

        if type == "semantic":
            self._semantic_chunking()
        elif type == "recursive":
            self._recursive_chunking()
        elif type == "fixed_size":
            self._fixed_size_chunking(self.chunk_size, self.chunk_overlap)
        else:

            raise ValueError(f"Unknown chunking type: {type}")
        
        
