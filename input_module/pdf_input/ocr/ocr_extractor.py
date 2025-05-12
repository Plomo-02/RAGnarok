import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import time
import re
import argparse
from pathlib import Path

class OCRPdfExtractor:
    def __init__(self, lang='ita+eng', dpi=300, tesseract_path=None):
        """
        Inizializza l'estrattore OCR
        
        Args:
            lang (str): Lingue da utilizzare con Tesseract (es. 'ita+eng')
            dpi (int): Risoluzione per la conversione PDF-immagine
            tesseract_path (str): Percorso all'eseguibile di Tesseract (solo per Windows)
        """
        self.lang = lang
        self.dpi = dpi
        
        # Configura il percorso di Tesseract se specificato (utile su Windows)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Verifica che Tesseract sia installato e funzionante
        try:
            pytesseract.get_tesseract_version()
            print(f"Tesseract OCR trovato. Versione: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            print(f"ERRORE: Tesseract OCR non trovato o non funzionante: {e}")
            print("Assicurati che Tesseract sia installato e, su Windows, specifica il percorso con tesseract_path")
    
    def extract_text_from_pdf(self, pdf_path, output_file=None, pages=None):
        """
        Estrae il testo da un PDF utilizzando OCR
        
        Args:
            pdf_path (str): Percorso al file PDF
            output_file (str, optional): Percorso per salvare il testo estratto
            pages (list, optional): Lista di numeri di pagina da elaborare (1-indexed)
                                   Se None, elabora tutte le pagine
        
        Returns:
            str: Il testo estratto
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Il file PDF non esiste: {pdf_path}")
        
        print(f"Convertendo il PDF in immagini (DPI={self.dpi})...")
        start_time = time.time()
        
        try:
            # Converti il PDF in immagini
            images = convert_from_path(
                pdf_path, 
                dpi=self.dpi,
                first_page=pages[0] if pages else None,
                last_page=pages[-1] if pages else None
            )
            
            print(f"Conversione completata: {len(images)} pagine ({time.time() - start_time:.2f} secondi)")
            print(f"Esecuzione OCR con Tesseract (lingua: {self.lang})...")
            
            # Estrai testo da ogni immagine
            full_text = ""
            for i, image in enumerate(images):
                page_num = (pages[i] if pages else i + 1)
                print(f"Elaborazione pagina {page_num}/{len(images)}...")
                
                page_start = time.time()
                page_text = pytesseract.image_to_string(image, lang=self.lang)
                
                # Pulisci il testo estratto (rimuovi caratteri non necessari)
                page_text = self.clean_text(page_text)
                
                # Aggiungi il testo della pagina al testo completo
                if page_text.strip():
                    full_text += f"\n\n--- Pagina {page_num} ---\n\n"
                    full_text += page_text
                
                print(f"  Completata in {time.time() - page_start:.2f} secondi")
            
            # Salva il testo in un file se richiesto
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                print(f"Testo salvato in: {output_file}")
            
            print(f"Estrazione OCR completata in {time.time() - start_time:.2f} secondi")
            return full_text
            
        except Exception as e:
            print(f"Errore durante l'estrazione OCR: {e}")
            return ""
    
    def clean_text(self, text):
        """
        Pulisce il testo estratto rimuovendo caratteri indesiderati
        e regolando la formattazione
        
        Args:
            text (str): Testo da pulire
        
        Returns:
            str: Testo pulito
        """
        # Rimuovi caratteri di controllo e mantieni solo spazi standard
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Normalizza gli spazi consecutivi
        text = re.sub(r' +', ' ', text)
        
        # Normalizza le interruzioni di riga
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def extract_from_image(self, image_path, output_file=None):
        """
        Estrae il testo da un'immagine utilizzando OCR
        
        Args:
            image_path (str): Percorso all'immagine
            output_file (str, optional): Percorso per salvare il testo estratto
            
        Returns:
            str: Il testo estratto
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Il file immagine non esiste: {image_path}")
        
        print(f"Esecuzione OCR sull'immagine: {image_path}")
        start_time = time.time()
        
        try:
            # Carica l'immagine
            image = Image.open(image_path)
            
            # Estrai il testo
            text = pytesseract.image_to_string(image, lang=self.lang)
            text = self.clean_text(text)
            
            # Salva il testo in un file se richiesto
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Testo salvato in: {output_file}")
            
            print(f"Estrazione OCR completata in {time.time() - start_time:.2f} secondi")
            return text
            
        except Exception as e:
            print(f"Errore durante l'estrazione OCR: {e}")
            return ""

if __name__ == "__main__":
    extractor = OCRPdfExtractor()
    
    # Determina se il file è un PDF o un'immagine
    file_ext = "C:/Users/Plomo/Desktop/ragnarok/RAGnarok/frontend/temp/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    if file_ext == '.pdf':
        extractor.extract_text_from_pdf(args.file, args.output, pages)
    elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif']:
        extractor.extract_from_image(args.file, args.output)
    else:
        print(f"Formato file non supportato: {file_ext}")
