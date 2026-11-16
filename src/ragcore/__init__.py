"""ragcore: a small, pluggable RAG baseline."""

from ragcore.config import RagConfig, build_rag, load_profile, register
from ragcore.pipeline.rag import Rag
from ragcore.types import Answer, Chunk, Citation, ConversationState, Document, Hit, Turn

__all__ = ["Answer", "Chunk", "Citation", "ConversationState", "Document", "Hit", "Rag", "RagConfig", "Turn",
           "build_rag", "load_profile", "register"]
