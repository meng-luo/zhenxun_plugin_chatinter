from .knowledge_rag_feedback import PluginRAGFeedbackMixin
from .knowledge_rag_retrieval import PluginRAGRetrievalMixin


class PluginRAGService(PluginRAGFeedbackMixin, PluginRAGRetrievalMixin):
    pass
