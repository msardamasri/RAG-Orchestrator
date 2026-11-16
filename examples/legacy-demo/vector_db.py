from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="documents", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim
        
        if not self.client.collection_exists(self.collection):
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
                print(f"Created collection: {self.collection}")
            except Exception as e:
                print(f"Collection might already exist: {e}")
                try:
                    info = self.client.get_collection(self.collection)
                    print(f"Using existing collection: {self.collection}")
                except:
                    raise

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points
        contexts = []
        sources = set()
        for result in results:
            payload = getattr(result, 'payload', None) or {}
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                contexts.append(text)
                sources.add(source)
        return {"contexts": contexts, "sources": list(sources)}