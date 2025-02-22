from db.session import with_session
from typing import Dict, List
import uuid
from db.models.llm_model import LLMConfigModel


@with_session
def delete_llm_config_from_db(session, model_name: str):
    """
    删除LLM模型配置
    """
    session.query(LLMConfigModel).filter(LLMConfigModel.model_name == model_name).delete()
    session.commit()
    return {"model": model_name}

@with_session
def add_llm_config_to_db(session, model_name: str, base_url: str, api_key: str):
    """
    新增LLM模型配置
    """
    m = LLMConfigModel(model_name=model_name, base_url=base_url, api_key=api_key)
    session.add(m)
    session.commit()
    return {"model": m.model_name, "base_url": m.base_url, "api_key": m.api_key}

@with_session
def list_llm_config_from_db(session) -> List[Dict]:
    """
    获取所有LLM模型配置
    """
    configs = session.query(LLMConfigModel).all()
    return [
        dict(
            model_name=config.model_name,
            base_url=config.base_url,
            api_key=config.api_key
        )
        for config in configs
    ]

