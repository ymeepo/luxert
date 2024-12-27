from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain

class Conversation:
    def __init__(self, model_path, retriever):
        self._retriever = retriever

        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
        )

        self._conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
        )

    def query(self, query, chat_history=None):
        if chat_history is None:
            chat_history = []
        response = self._conversation({"question": query, "chat_history": chat_history})
        answer = response["answer"]
        return answer
    
    def get_conversation(self):
        return self._conversation

