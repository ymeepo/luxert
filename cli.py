from luxert_core.indexer import Indexer
from luxert_core.conversation import Conversation
from luxert_cli.config import *
from langchain_core.documents import Document

def main():
    print("Starting up...")
    print("Initializing indexer...")
    indexer = Indexer(persist_dir=CHROMA_DIR)

    print("Adding document...")
    document = Document(page_content="An engineering manager works at Google and does engineering things.")
    indexer.index_document(document)

    print("Adding raw documents...")
    indexer.index_files("luxert_data/raw_documents")

    print("Initiatizing conversation...")
    conversation = Conversation(
        model_path=MODELS_DIR + "/" + META_LLAMA_3_8B_INSTRUCT_Q5_K_M_GGUF,
        retriever=indexer.get_retriever()
    )
    
    chat_history = []
    while(True):
        user_query = input("Ask a question: ")
        if (user_query == "exit"):
            break
        agent_response = conversation.query(user_query, chat_history)
        # Update chat history with first exchange
        chat_history.append((f"User: {user_query}", f"Agent: {agent_response}"))
        print("Response: " + agent_response)

if __name__ == "__main__":
    main()
