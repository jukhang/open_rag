#! python3
# -*- encoding: utf-8 -*-
'''
@File    : milvus.py
@Time    : 2024/06/13 16:54:09
@Author  : longfellow
@Version : 1.0
@Email   : longfellow.wang@gmail.com
'''


from typing import List, Dict
from pymilvus import (
    connections,
    DataType,
    Collection,
    FieldSchema,
    MilvusClient,
    MilvusException,
    CollectionSchema,
)
from loguru import logger


class MilvusStore():
    """
    Milvus 存储 查询
    """

    @staticmethod
    def create_schema(dim: int = 1024) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
            FieldSchema(name="kb_name", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        return CollectionSchema(fields=fields, description="Chunk Vector Collection.")
    
    def __init__(self, host, port) -> None:
        self.client = MilvusClient(f"http://{host}:{port}")
        connections.connect(host=host, port=port)

    def has_collection(self, collection_name) -> bool:
        return self.client.has_collection(collection_name)
    
    def detele_collection(self, collection_name):
        if self.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        else:
            logger.info(f"Collection {collection_name} does not exist")
    
    def get_collection(self, collection_name) -> Collection:
        has = self.client.has_collection(collection_name)
        if has:
            return Collection(collection_name)
        else:
            raise MilvusException(
                code=404, message=f"Collection {collection_name} already exists")

    def list_collections(self) -> List[str]:
        return self.client.list_collections()

    def create_collection(self, collection_name, dim, metric_type="L2"):
        has = self.client.has_collection(collection_name)
        if has:
            logger.info(f"Collection {collection_name} already exists")
        else: 
            schema = MilvusStore.create_schema(dim)
            index_params = self.client.prepare_index_params("vector", metric_type=metric_type)

            return self.client.create_collection(collection_name, schema=schema, index_params=index_params)

    def insert(self, collection_name: str, entities: List):
        if not self.has_collection(collection_name):
            raise MilvusException(
                code=404, message=f"Collection {collection_name} does not exist"
            )
        self.client.insert(collection_name, entities)

    def insert_entities_with_collection(self, collection: Collection, entities: List):
        collection.insert(entities)

    def query_entities(self, collection_name: str, filter: str, top_k: int) -> List[Dict]:
        if not self.has_collection(collection_name):
            raise MilvusException(
                code=404, message=f"Collection {collection_name} does not exist"
            )

        return self.client.query(collection_name, filter=filter, top_k=top_k)
    

    def delete_entry_with_kb_name(self, collection_name: str, kb_name: str):
        if not self.has_collection(collection_name):
            raise MilvusException(
                code=404, message=f"Collection {collection_name} does not exist"
            )

        collection = Collection(name=collection_name)
        collection.load()
        result = collection.delete(expr=f"kb_name == '{kb_name}'")
        
        return result

    
    def delete_entry_with_file_name(self, collection_name: str, file_name: str, kb_name: str):
        if not self.has_collection(collection_name):
            raise MilvusException(
                code=404, message=f"Collection {collection_name} does not exist"
            )

        collection = Collection(name=collection_name)
        collection.load()
        result = collection.delete(expr=f"file_name == '{file_name}' and kb_name == '{kb_name}'")
        
        return result

    def delete_entry_with_uuid(self, collection_name: str, chunk_uuid: str):
        if not self.has_collection(collection_name):
            raise MilvusException(
                code=404, message=f"Collection {collection_name} does not exist"
            )

        collection = Collection(name=collection_name)
        collection.load()
        result = collection.delete(expr=f"id == '{chunk_uuid}'")
        
        return result

    """
    TODO: 布尔表达式
        https://www.milvus-io.com/boolean
    """
    def search(self, 
               collection_name: str,
               kb_names: List[str],
               query : List,
               top_k: int = 3,
               consistency_level: str = "Eventually") -> List[Dict]:
        if not self.has_collection(collection_name):
            raise MilvusException(
                code=404, message=f"Collection {collection_name} does not exist"
            )

        collection = self.get_collection(collection_name)
        collection.load()
        search_param = {
            "expr": f"kb_name in {kb_names}",
            "data": [query],
            "anns_field": "vector",
            "param": {"metric_type": "L2", "params": {"nprobe": 10}},
            "limit": top_k,
            "output_fields": ["id", "kb_name", "file_name", "chunk"],
            "consistency_level": consistency_level
        }

        results = collection.search(**search_param)
        return [{
            "id": r.to_dict().get("entity").get("id"),
            "kb_name": r.to_dict().get("entity").get("kb_name"),
            "file_name": r.to_dict().get("entity").get("file_name"),
            "chunk": r.to_dict().get("entity").get("chunk")
        } for result in results for r in result]


from settings import settings

milvus_client = MilvusStore(settings.MILVUS_HOST, settings.MILVUS_PORT)
