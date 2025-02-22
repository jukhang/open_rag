from sqlalchemy import Column, Integer, String, DateTime, JSON, func

from db.base import Base


class LLMConfigModel(Base):
    """
    LLM模型配置
    """
    __tablename__ = 'llm_config'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    model_name = Column(String(50), comment='模型名称')
    base_url = Column(String(255), comment='模型地址')
    api_key = Column(String(255), comment='api-key')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        return f"<llm_config(id='{self.id}', model_name='{self.model_name}', model_type='{self.model_type}', model_path='{self.model_path}', create_time='{self.create_time}')>"    