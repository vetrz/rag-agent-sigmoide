class RagAgent:
    def __init__(self):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma

            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device":"cpu"}
            )

            self.vector_store = Chroma(
                persist_directory="./rag-agent-sigmoide/chroma_db", 
                embedding_function=embeddings
            )

            self.retriever = self.vector_store.as_retriever()
            print("ChromaDB loaded.")

        except Exception:
            print("ChromaDB not found. Creat a new database")
            self.database()

    def database(self):
        from bs4 import BeautifulSoup
        import os
        import re

        complete_text: list = []
        root_directory = "."

        for current_path, _, files in os.walk(root_directory):
            for filename in files:
                if filename.lower().endswith(".svg"):
                    complet_path = os.path.join(current_path, filename)        
                    content_svg = ""
                try:
                    with open(complet_path, "r", encoding="utf-8") as f:
                        content_svg = f.read()
                    soup = BeautifulSoup(content_svg, "xml")

                    text_elements = soup.find_all("text")
                    for text_element in text_elements:
                        text = text_element.get_text()
                        text = re.sub(r'^\s*$\n', "", text, flags=re.MULTILINE)
                        complete_text.append(text)

                except Exception as e:
                    print(f"error when reading {complet_path}: {e}")

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=10,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.create_documents(complete_text)

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device":"cpu"}
        )

        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        self.retriever = vector_store.as_retriever()

    def question_and_answer(self, choice: int, query: str) -> print:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        
        match choice:
            case 1:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from dotenv import load_dotenv
                load_dotenv()
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            case 2:
                from langchain_ollama import OllamaLLM
                llm = OllamaLLM(model="llama3")
            case _:
                return print("invalid model selection")

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use the following context to answer the question.
        If you don't know the answer, say that you don't have enough information in the context.

        Context:
        {context}

        Asking: {input}
        """)

        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
        )

        rag_chain = create_retrieval_chain(self.retriever, document_chain)

        response = rag_chain.invoke({"input": query})

        return print(f"\nresponse: {response['answer']}")

if __name__ == "__main__":
    selection = int(input("""
Choice your model:
    1 - gemini-2.5-flash 
        Note : gemini_api_key must be in a .env file or in a environment variable.
  
    2 - Ollama3
        Note: You need to download the Ollama3 template.
          
Valuer: """))
    
    question = input("""
Type your Question: """)

    agent = RagAgent()

    agent.question_and_answer(choice=selection, query=question)