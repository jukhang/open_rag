from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func, Text

from db.base import Base


class SummaryChunkModel(Base):
    """
    chunk summary模型，用于存储file_doc中每个doc_id的chunk 片段，
    数据来源:
        用户输入: 用户上传文件，可填写文件的描述，生成的file_doc中的doc_id，存入summary_chunk中
        程序自动切分 对file_doc表meta_data字段信息中存储的页码信息，按每页的页码切分，自定义prompt生成总结文本，将对应页码关联的doc_id存入summary_chunk中
    后续任务:
        矢量库构建: 对数据库表summary_chunk中summary_context创建索引，构建矢量库，meta_data为矢量库的元数据（doc_ids）
        语义关联： 通过用户输入的描述，自动切分的总结文本，计算
        语义相似度

    """
    __tablename__ = 'summary_chunk'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    kb_name = Column(String(50), comment='知识库名称')
    file_name = Column(String(255), comment='文件名称')
    chunk_id = Column(String(255), comment='切片序号')
    chunk = Column(Text, comment='文件切片')
    chunk_uuid = Column(String(255), comment="文件切片uuid")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        return (f"<SummaryChunk(id='{self.id}', kb_name='{self.kb_name}', summary_context='{self.file_name}',"
                f" chunk_id='{self.chunk_id}', chunk='{self.chunk}')> chunk_uuid='{self.chunk_uuid}'")
