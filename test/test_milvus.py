from db.milvus import milvus_client



# milvus_client.delete_entry_with_file_name(collection_name="langchat", file_name="西安迅腾公司简介.md")


milvus_client.delete_entry_with_kb_name(collection_name="langchat", kb_name="天天测试")