#! python3
# -*- encoding: utf-8 -*-
"""
@File    : es.py
@Time    : 2024/06/13 17:13:39
@Author  : longfellow
@Version : 1.0
@Email   : longfellow.wang@gmail.com
"""


from typing import List
from elasticsearch import Elasticsearch, helpers
from loguru import logger

class ESClient():
    mappings = {
        "settings": {
            "index": {"similarity": {"default": {"type": "BM25", "b": 0.2, "k1": 1.5}}}
        },
        "mappings": {
            "properties": {
                "kb_name": {"type": "keyword"},
                "file_name": {"type": "keyword"},
                "chunk_uuid": {"type": "keyword"},
                "chunk": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart",
                },
                "metadata": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "created_at": {"type": "date"},
            }
        },
    }

    def __init__(self, host="localhost", port="9200", size=3) -> None:
        self.size = size
        self.min_score = 0.2
        self.es = Elasticsearch(f"http://{host}:{port}")

    def index_exists(self, index_name: str) -> bool:
        return self.es.indices.exists(index=index_name)

    def create_index(self, index_name: str):
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=ESClient.mappings)
        else:
            logger.info(f"Index '{index_name}' already exists.")

    def insert(self, index_name, chunks):
        def gendata(chunks):
            for chunk in chunks:
                yield {"_index": index_name, "_source": chunk}

        helpers.bulk(self.es, gendata(chunks=chunks))

    def search(self, index_name: str, kb_names: List[str], query):
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"chunk": query}},
                        {"terms": {"kb_name": kb_names}},
                    ]
                }
            },
            "size": self.size,
            "min_score": self.min_score,
        }

        response = self.es.search(
            index=index_name,
            body=query_body,
        )

        return response["hits"]["hits"]

    # 删除索引
    def delete_index(self, index_name):
        try:
            if self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
            else:
                logger.info(f"Index '{index_name}' does not exist.")
        except Exception as e:
            logger.info(f"Failed to delete index '{index_name}': {e}")

    def detele_record_with_uuid(self, index_name, chunk_uuid):
        # 构造查询条件，匹配指定的 chunk_uuid
        query = {
            "query": {
                "term": {
                    "chunk_uuid": chunk_uuid
                }
            }
        }

        response = self.es.delete_by_query(index=index_name, body=query)
        logger.info(f"Deleted {response['deleted']} records with chunk_uuid: {chunk_uuid}")
        return response


from settings import settings

es_client = ESClient(host=settings.ES_HOST, port=settings.ES_PORT)
