from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func

from db.base import Base


class KnowledgeFileModel(Base):
    """
    知识文件模型
    """
    __tablename__ = 'knowledge_file'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='知识文件ID')
    kb_name = Column(String(50), comment='所属知识库名称')
    file_name = Column(String(255), comment='文件名')
    file_ext = Column(String(10), comment='文件扩展名')
    is_deleted = Column(Boolean, default=False, comment='是否删除')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        return f"<KnowledgeFile(id='{self.id}', file_name='{self.file_name}', file_ext='{self.file_ext}', kb_name='{self.kb_name}', create_time='{self.create_time}')>"


class FileDocModel(Base):
    """
    文件-向量库文档模型
    """
    __tablename__ = 'file_doc'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    kb_name = Column(String(50), comment='知识库名称')
    file_name = Column(String(255), comment='文件名称')
    doc_id = Column(String(50), comment="向量库文档ID")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        return f"<FileDoc(id='{self.id}', kb_name='{self.kb_name}', file_name='{self.file_name}', doc_id='{self.doc_id}', metadata='{self.meta_data}')>"
