from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from RAGnarok.input_module.pdf_input.read import get_input_path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkManager:
    def __init__(self, chunk_size = 1000, chunk_overlap  = 0.2,breakpoint_threshold_type = "percentile"):
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def semantic_chunking(self):
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
        
    def recursive_chunking(self):
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


    
        

        