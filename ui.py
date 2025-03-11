import streamlit as st
import os
from input_module.pdf_input.read import extract_text_from_pdf

def main():
    # Titolo dell'app
    st.title("Carica un file per l'elaborazione")

    # Widget per il caricamento del file
    uploaded_file = st.file_uploader("Trascina o seleziona un file", type=["txt", "csv", "json", "pdf"])  # Specifica i formati supportati

    if uploaded_file is not None:
        # Mostra il nome del file caricato
        st.success(f"File caricato: {uploaded_file.name}")
        
        # Salvataggio temporaneo
        save_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info(f"File salvato temporaneamente in {save_path}")

        extract_text_from_pdf(save_path)
        
        # Aggiungi qui la tua logica di elaborazione
        st.write("🔄 Elaborazione del file in corso...")
        
        # Esempio di elaborazione: lettura del contenuto se è un file di testo
        if uploaded_file.type == "text/plain":
            with open(save_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.text_area("Contenuto del file:", content, height=200)
        
        st.success("✅ Elaborazione completata!")

if __name__ == "__main__":
    main()
