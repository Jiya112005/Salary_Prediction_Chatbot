from langchain_ollama import OllamaEmbeddings #text to vector conversion
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd 

# data loading
df = pd.read_csv("salary_text_description.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids=[]
    
    for i,row in df.iterrows():
        document = Document(
            page_content=row["Job Title"] +" "+row["Skills"]+" "+row["Next_Role"]+" "+row["text_description"]+" "+str(row["Salary"]),
            metadata = {"experience":row["Years of Experience"],"age":row["Age"]} ,# additional information 
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
    
    # adding to vector store
vector_store = Chroma(
    collection_name = "salary_explanations",
    persist_directory=db_location,
    embedding_function=embeddings
)
    
if add_documents:
    vector_store.add_documents(documents=documents,ids=ids)
    
    # make this vector database be usable connected to llm
retriever = vector_store.as_retriever(
    search_kwargs={'k':5}
)

print(f"Running vector.py successfully/ Overwrite the {db_location} ")