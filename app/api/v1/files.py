import os
import json
import uuid
import asyncio
from typing import List
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from fastapi import APIRouter, UploadFile, File, Form

from libs.utils import run_in_thread_pool, get_file_path
from db.milvus import milvus_client


from libs.embedding import embedding_function
from db.repository.conversation_repository import add_conversation_to_db
from db.repository.conversation_repository import list_conversations_from_db
from db.repository.message_repository import add_message_to_db
from db.repository.knowledge_metadata_repository import (
    get_chunk_with_uuid, get_chunk_with_id, get_file_all_chunks
)
from db.repository.knowledge_metadata_repository import list_summary_from_db
from db.repository.knowledge_file_repository import list_files_from_db
from db.repository.knowledge_file_repository import delete_file_from_db
from db.repository.knowledge_metadata_repository import delete_chunk_from_db
from db.repository.knowledge_base_repository import delete_kb_from_db
from db.repository.knowledge_file_repository import delete_files_from_db
from db.repository.knowledge_metadata_repository import delete_summary_from_db
from db.repository.knowledge_base_repository import add_kb_to_db
from db.repository.knowledge_base_repository import list_kbs_from_db

from db.milvus import milvus_client
from libs.llm import default_model
from libs.utils import validate_kb_name
from settings import settings

from loguru import logger

router = APIRouter()

def _save_files_in_thread(files: List[UploadFile], kb_name: str, override: bool):
    def save_file(file: UploadFile, kb_name: str, override: bool) -> dict:
        '''
        保存单个文件。
        '''
        try:
            filename = file.filename
            file_path = get_file_path(kb_name=kb_name, doc_name=filename)
            data = {"kb_name": kb_name, "file_name": filename}

            file_content = file.file.read()  # 读取上传文件的内容
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                return dict(code=404, msg=f"文件 {filename} 已存在。", data=data)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            return dict(code=500, msg=f"{filename} 文件上传失败，报错信息为: {e}", data=data)

    params = [{"file": file, "kb_name": kb_name, "override": override} for file in files]

    for result in run_in_thread_pool(save_file, params=params):
        yield result


@router.post("/init", summary="初始化接口")
async def init() -> JSONResponse:
    try:
        from db.migrate import reset_tables
        reset_tables()

        if milvus_client.has_collection("langchat"):
            milvus_client.detele_collection("langchat")
        milvus_client.create_collection("langchat", 1024)

        return JSONResponse(dict(code=200, msg="数据库初始化成功", data={}))
    except:
        return JSONResponse(dict(code=500, msg="数据库初始化失败", data={}))


from app.tasks import celery_process_files
@router.post("/upload", summary="批量上传文件接口")
async def upload_files(
    files: List[UploadFile] = File(..., description="批量上传文件接口"),
    kb_name: str = Form("default", description="知识库名称", examples=[""]),
    override: bool = Form(True, description="覆盖已有文件"),
    # background_tasks: BackgroundTasks = BackgroundTasks(),
) -> JSONResponse:
    if not validate_kb_name(kb_name):
        return JSONResponse(dict(code=400, msg="知识库名称不合法", data=[]))

    knowledge_base = f"knowledge_base/{kb_name}"
    os.makedirs(knowledge_base, exist_ok=True)

    success_files = {}
    failed_files = {}

    # 先将上传的文件保存到磁盘
    for result in _save_files_in_thread(files, kb_name=kb_name, override=override):
        filename = result["data"]["file_name"]
        if result["code"] == 200:
            success_files[filename] = result["msg"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

    # 将文件解析任务放到后台执行
    # background_tasks.add_task(process_files, files, kb_name)
    task = celery_process_files.delay(success_files.keys(), kb_name)

    return JSONResponse(
        dict(
            code=200,
            msg="文件上传成功，正在后台解析",
            data=dict(
                kb_name=kb_name, 
                files=success_files, 
                failed_files=failed_files,
                task_id=task.id,
            ),
        )
    )


@router.get("/task", summary="获取任务状态接口")
async def get_task_status(task_id: str) -> JSONResponse:
    task = celery_process_files.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return JSONResponse(
            dict(
                code=200,
                msg="success",
                data=dict(task_id=task_id, status=task.result),
            )
        )
    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=dict(task_id=task_id, status=task.status),
        )
    )

class KnowledgeBaseInfo(BaseModel):
    kb_name: str
    kb_info: str

@router.post("/knowledge_base", summary="创建知识库接口")
async def create_knowledge_base(kb_info: KnowledgeBaseInfo) -> JSONResponse:
    if not validate_kb_name(kb_info.kb_name):
        return JSONResponse(dict(code=400, msg="知识库名称不合法", data=[]))
    
    knowledge_base = f"knowledge_base/{kb_info.kb_name}"
    os.makedirs(knowledge_base, exist_ok=True)

    add_kb_to_db(kb_name=kb_info.kb_name, kb_info=kb_info.kb_info, vs_type='default', embed_model='default')

    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=dict(kb_name=kb_info.kb_name, kb_info=kb_info.kb_info),
        )
    )


@router.get("/knowledge_base", summary="获取知识库列表接口")
async def get_knowledge_base() -> JSONResponse:
    kbs = list_kbs_from_db()
    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=kbs,
        )
    )

@router.delete("/knowledge_base", summary="删除知识库接口")
async def delete_knowledge_base(kb_name: str) -> JSONResponse:
    delete_kb_from_db(kb_name=kb_name)
    delete_files_from_db(kb_name=kb_name)
    delete_summary_from_db(kb_name=kb_name)

    milvus_client.delete_entry_with_kb_name(collection_name='langchat', kb_name=kb_name)

    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=dict(kb_name=kb_name),
        )
    )


class DelFileInfo(BaseModel):
    kb_name: str
    file_name: str

@router.post("/delete/file", summary="删除知识库文件接口")
async def delete_file(del_file: DelFileInfo) -> JSONResponse:
    '''
        1. 在知识库文件表中删除记录
        2. 在切片表中删除记录
        3. 在milvus中删除记录
    '''
    kb_name = del_file.kb_name
    file_name = del_file.file_name

    delete_file_from_db(kb_name=kb_name, file_name=file_name)
    delete_chunk_from_db(kb_name=kb_name, file_name=file_name)
    milvus_client.delete_entry_with_file_name(
        collection_name='langchat', file_name=file_name, kb_name=kb_name
    )

    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=dict(kb_name=kb_name, file_name=file_name),
        )
    )

@router.get("/files", summary="获取知识库所有文件列表接口")
async def get_knowledge_base(kb_name: str) -> JSONResponse:
    kbs = list_files_from_db(kb_name=kb_name)
    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=kbs,
        )
    )

@router.get("/chunk", summary="获取知识库所有文件列表接口")
async def get_knowledge_base(kb_name: str, file_name: str) -> JSONResponse:
    kbs = list_summary_from_db(kb_name=kb_name, file_name=file_name)
    
    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data={
                "kb_name": kb_name,
                "file_name": file_name,
                "chunks": kbs,
            },
        )
    )


class ConversationInfo(BaseModel):
    chat_type: str = "chat"
    name: str = "新会话"

@router.post("/conversation", summary="创建会话接口")
async def create_conversation(conversation_info: ConversationInfo) -> JSONResponse:
    '''创建新的会话'''
    conversation_id = str(uuid.uuid4())
    add_conversation_to_db(
        chat_type=conversation_info.chat_type,
        name=conversation_info.name,
        conversation_id=conversation_id
    )
    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=dict(conversation_id=conversation_id),
        )
    )


@router.get("/conversation", summary="获取会话列表接口")
async def get_conversation_list() -> JSONResponse:
    conversations = list_conversations_from_db()
    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=conversations,
        )
    )

@router.get("/messages", summary="获取会话列表接口")
async def get_all_messages_with_conversation_uuid(
    conversation_id: str
) -> JSONResponse:
    from db.repository.message_repository import filter_message

    messages = filter_message(conversation_id=conversation_id)
    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=messages,
        )
    )


class LLMModelConfig(BaseModel):
    model_name: str
    api_key: str
    base_url: str

model_cache = {}

@router.get("/llm/model", summary="获取模型列表接口")
async def get_llm_model_list() -> JSONResponse:
    from db.repository.llm_repository import list_llm_config_from_db
    llm_configs = list_llm_config_from_db()

    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=llm_configs,
        )
    )

@router.delete("/llm/model", summary="删除模型接口")
async def delete_llm_model(model_name: str) -> JSONResponse:
    from db.repository.llm_repository import delete_llm_config_from_db
    delete_llm_config_from_db(model_name=model_name)

    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=dict(model_name=model_name),
        )
    )

from libs.llm import LLM
@router.post("/llm/model", summary="创建模型接口")
async def create_llm_model(llm_model_config: LLMModelConfig) -> JSONResponse:
    '''创建新的模型'''
    from db.repository.llm_repository import add_llm_config_to_db

    add_llm_config_to_db(
        model_name=llm_model_config.model_name,
        api_key=llm_model_config.api_key,
        base_url=llm_model_config.base_url
    )

    model = LLM(
        model=llm_model_config.model_name,
        api_key=llm_model_config.api_key,
        base_url=llm_model_config.base_url
    )
    model_cache[llm_model_config.model_name] = model
    return JSONResponse(
        dict(
            code=200,
            msg="success",
            data=dict(model_name=llm_model_config.model_name),
        )
    )

def generate_llm_model():
    from db.repository.llm_repository import list_llm_config_from_db
    llm_configs = list_llm_config_from_db()
    for llm_config in llm_configs:
        model = LLM(
            model=llm_config.get("model_name"),
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url")
        )
        model_cache[llm_config.get("model_name")] = model
    

class ChatInfo(BaseModel):
    question: str
    conversation_id: str = None
    model: str = "default"
    kb_name: str = None
    file_name: str = None

@router.post("/chat", summary="创建知识库接口")
async def llm_chat(chat_info: ChatInfo) -> JSONResponse:
    '''创建新的会话'''
    if chat_info.conversation_id is None:
        conversation_id = str(uuid.uuid4())
        name = chat_info.question[0:50]
        chat_type = 'chat'
        if chat_info.kb_name is not None or chat_info.file_name is not None:
            chat_type = 'rag_chat'

        add_conversation_to_db(chat_type=chat_type, name=name, conversation_id=conversation_id)
    
    documents = {}
    document_chunks_id = {}
    prompt_douments = {}
    if chat_info.kb_name is not None and chat_info.file_name is None:
        query = embedding_function([chat_info.question])[0]

        vector_search = milvus_client.search('langchat', [chat_info.kb_name], query)
        for result in vector_search:
            chunk = get_chunk_with_uuid(result['id'])
            prev_chunk, next_chunk = None, None
            if f"{chunk['id']-1}-{chunk['file_name']}" not in documents.keys() and chunk['id'] > 0:
                prev_chunk = get_chunk_with_id(chunk['id']-1, chunk['file_name'], chunk['kb_name'])
            if f"{chunk['id']+1}-{chunk['file_name']}" not in documents.keys():
                next_chunk = get_chunk_with_id(chunk['id']+1, chunk['file_name'], chunk['kb_name'])

            if chunk is not None:
                documents[f"{chunk['id']}-{chunk['file_name']}"] = chunk
                document_chunks_id.setdefault(chunk['file_name'], []).append(chunk['id'])
            if prev_chunk is not None:
                documents[f"{prev_chunk['id']}-{prev_chunk['file_name']}"] = prev_chunk
                document_chunks_id.setdefault(prev_chunk['file_name'], []).append(prev_chunk['id'])
            if next_chunk is not None:
                documents[f"{next_chunk['id']}-{next_chunk['file_name']}"] = next_chunk
                document_chunks_id.setdefault(next_chunk['file_name'], []).append(next_chunk['id'])
        
        for file_name, chunk_ids in document_chunks_id.items():
            for chunk_id in sorted(set(chunk_ids)):
                prompt_douments.setdefault(
                    file_name, []
                ).append(
                    dict(
                        id=chunk_id,
                        chunk=documents[f"{chunk_id}-{file_name}"]['chunk']
                    )
                )

    if chat_info.file_name is not None:
        '''文件对话'''
        chunks = get_file_all_chunks(kb_name=chat_info.kb_name, file_name=chat_info.file_name)

        prompt_douments[chat_info.file_name] = chunks
    
    quote = []
    background_knowledge = ''
    for file_name, chunks in prompt_douments.items():
        quote.append({
            "file_name": file_name,
            "url": f'http://{settings.APP_HOST}:{settings.APP_PORT}files/{chat_info.kb_name}/{file_name}'
        })
        background_knowledge += f"## {file_name}\n"
        for chunk in chunks:
            background_knowledge += f"{chunk['chunk']}\n"
        background_knowledge += '\n'
    
    if background_knowledge == '':
        system_prompt = '你是一个智能助手，请用简练的语言回答用户的问题'
        question = chat_info.question
    else:
        system_prompt = f"你是一个智能助手，请结合用户提供的背景知识，回答问题"
        question = f"==背景知识==\n{background_knowledge}\n\n==问题=={chat_info.question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    llm_model: LLM = model_cache.get(chat_info.model, default_model)
    resp = llm_model.sse_invoke(messages)
    async def sse_stream():
        answer = ''
        for delta_content in resp:
            answer += delta_content["content"]
            yield json.dumps(delta_content, ensure_ascii=False)
            await asyncio.sleep(0)
        
        _id = add_message_to_db(conversation_id=chat_info.conversation_id, chat_type='chat', query=chat_info.question,response=answer)

        if quote:
            yield json.dumps(quote, ensure_ascii=False)
        
        
            
    return EventSourceResponse(sse_stream())