from db.session import with_session
import uuid
from db.models.conversation_model import ConversationModel


@with_session
def add_conversation_to_db(session, chat_type, name="", conversation_id=None):
    """
    新增聊天记录
    """
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    c = ConversationModel(id=conversation_id, chat_type=chat_type, name=name)

    session.add(c)
    return c.id


@with_session
def list_conversations_from_db(session):
    """
    列出所有聊天记录
    """
    results = session.query(ConversationModel).all()
    return [{"id": result.id, "name": result.name, } for result in results]