from db.models.knowledge_metadata_model import SummaryChunkModel
from db.session import with_session
from typing import List, Dict

from sqlalchemy import and_


@with_session
def get_file_all_chunks(session, kb_name: str, file_name: str) -> int:
    chunks = session.query(SummaryChunkModel).filter(
        and_(
            SummaryChunkModel.kb_name.ilike(kb_name),
            SummaryChunkModel.file_name.ilike(file_name)
        )
    ).all()
    return [
        dict(
            id=chunk.chunk_id,
            chunk=chunk.chunk,
            # chunk_uuid=chunk.chunk_uuid,
            # file_name=chunk.file_name,
        )
        for chunk in chunks
    ]


@with_session
def get_chunk_with_uuid(session, chunk_uuid: str) -> Dict:
    doc = session.query(SummaryChunkModel).filter(SummaryChunkModel.chunk_uuid.ilike(chunk_uuid)).first()


    
    if doc is None:
        return None
    return {"id": int(doc.chunk_id), "chunk": doc.chunk, "chunk_uuid": doc.chunk_uuid, 'file_name': doc.file_name, 'kb_name': doc.kb_name}

@with_session
def get_chunk_with_id(session, chunk_id: int, file_name: str, kb_name: str) -> Dict:
    doc = session.query(SummaryChunkModel).filter(
        and_(
            SummaryChunkModel.chunk_id == chunk_id,
            SummaryChunkModel.kb_name.ilike(kb_name),
            SummaryChunkModel.file_name.ilike(file_name)
        )
    ).first()
    
    
    if doc is None:
        return None
    return {"id": int(doc.chunk_id), "chunk": doc.chunk, "chunk_uuid": doc.chunk_uuid, 'file_name': doc.file_name, 'kb_name': doc.kb_name}

@with_session
def list_summary_from_db(session,
                         kb_name: str,
                         file_name: str,
                         ) -> List[Dict]:
    docs = session.query(SummaryChunkModel).filter(
        and_(
            SummaryChunkModel.kb_name.ilike(kb_name), 
            SummaryChunkModel.file_name.ilike(file_name)
        )
    )

    return [{"id": doc.chunk_id, "chunk": doc.chunk, "chunk_uuid": doc.chunk_uuid} for doc in docs.all()]


@with_session
def delete_chunk_from_db(session,
                         kb_name: str,
                         file_name: str,
                         ) -> bool:
    '''删除某个文件的所有chunk'''
    query = session.query(SummaryChunkModel).filter(
        and_(
            SummaryChunkModel.kb_name.ilike(kb_name),
            SummaryChunkModel.file_name.ilike(file_name)
        )
    )
    query.delete(synchronize_session=False)
    session.commit()
    return True


@with_session
def delete_summary_from_db(session,
                           kb_name: str
                           ) -> List[Dict]:
    '''
    删除知识库chunk summary，并返回被删除的Dchunk summary。
    返回形式：[{"id": str, "summary_context": str, "doc_ids": str}, ...]
    '''
    # docs = list_summary_from_db(kb_name=kb_name)
    query = session.query(SummaryChunkModel).filter(SummaryChunkModel.kb_name.ilike(kb_name))
    query.delete(synchronize_session=False)
    session.commit()
    return True


@with_session
def add_summary_to_db(
    session,
    kb_name: str,
    file_name: str,
    chunk_id: int,
    chunk: str,
    chunk_uuid: str
) -> bool:
    '''
    将总结信息添加到数据库。
    summary_infos形式：[{"summary_context": str, "doc_ids": str}, ...]
    '''
    obj = SummaryChunkModel(
        kb_name= kb_name,
        file_name= file_name,
        chunk_id= chunk_id,
        chunk= chunk,
        chunk_uuid= chunk_uuid,
    )
    session.add(obj)
    return True


@with_session
def count_summary_from_db(session, kb_name: str) -> int:
    return session.query(SummaryChunkModel).filter(SummaryChunkModel.kb_name.ilike(kb_name)).count()
