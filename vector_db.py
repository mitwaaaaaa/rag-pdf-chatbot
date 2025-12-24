#it will allow us to connect to our qdrant database
# we can search something in db
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self,url="http://localhost:6333",collection="docs",dim=384):
        self.client = QdrantClient(url,timeout=30) #if won't connect in 30s, it will crash the program
        self.collection = collection
        #if collection doesnt exist
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name = self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE) #calculating distance of a point different points from our database
            )
    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)
        # it will use this and create a point structure
        #insert it in db
        #series of ids -> converted into vectors
        #payload is the real data, vectorized the information that we need


    #searching for vector
    def search(self, query_vector, top_k=5):
        results = self.client.search(
            collection_name = self.collection,
            query_vector =  query_vector,
            with_payload = True,
            limit = top_k # we are looking for these many results from database
        )

        contexts = []
        sources = set() #from which docs we are pulling out the info
        #set ds because we don't want to store duplicate source

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source","")
            if text:
                contexts.append(text)
                sources.add(source)
        return {"contexts": contexts, "sources": list(sources)}

