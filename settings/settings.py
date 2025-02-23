#! python3
# -*- encoding: utf-8 -*-
'''
@File    : settings.py
@Time    : 2025/02/18 11:12:32
@Author  : longfellow 
@Email   : longfellow.wang@gmail.com
@Version : 1.0
@Desc    : None
'''



import os
from pydantic_settings import BaseSettings

KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
os.makedirs(KB_ROOT_PATH, exist_ok=True)
DB_PATH = os.path.join(KB_ROOT_PATH, "info.db")

class Settings(BaseSettings):
    SQLALCHEMY_DATABASE_URI: str = f"sqlite:///{DB_PATH}"

    APP_HOST: str
    APP_PORT: int = 9009

    MILVUS_HOST: str
    MILVUS_PORT: int = 19530

    XINFERENCE_HOST: str
    XINFERENCE_PORT: int = 9997

    XINFERENCE_API_KEY: str = "NOT NULL"
    EMBEDDING_MODEL: str = "bge-m3" # 1024 dim, 8192 tokens
    RERANK_MODEL: str = "bge-reranker-v2-m3"

    DEFAULT_API_KEY: str = ''
    DEFAULT_BASE_URL: str = ''
    DEFAULT_MODEL: str = ''

    MINERU_HOST: str = ""
    MINERU_PORT: int = 8888

    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    class Config:
        env_prefix = ""
        env_file = ".env"

settings = Settings()