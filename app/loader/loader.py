import uuid
import asyncio
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import UploadFile 

from libs.utils import get_file_path
from db.milvus import milvus_client
import aiohttp
from settings import settings
from libs.embedding import embedding_function
from db.repository.knowledge_file_repository import add_docs_to_db
from db.repository.knowledge_metadata_repository import add_summary_to_db



async def pdf2docs(kb_name: str, file_name: str) -> dict:
    file_path = get_file_path(kb_name=kb_name, doc_name=file_name)
    
    url = f"http://{settings.MINERU_HOST}:{settings.MINERU_PORT}/pdf_parse"
    
    params = {
        "parse_method": "auto",
        "is_json_md_dump": "true"
    }
    from aiohttp import FormData
    # 使用 FormData 构造符合 FastAPI 期望的文件上传格式
    data = FormData()
    data.add_field(
        name="pdf_file",
        value=open(file_path, 'rb'),
        filename=file_name,
        content_type='application/pdf'
    )
    # 使用 aiohttp 发送异步请求
    async with aiohttp.ClientSession() as session:
        async with session.post(url, params=params, data=data) as response:
            if response.status == 200:
                try:
                    contents = await response.json()
                    text = ''
                    for content in contents['content']:
                        text += content.get('text', '')
                        text += '\n\n'
                    return {"file_name": file_name, "text": text}
                except Exception as e:
                    raise RuntimeError(f"解析文件{file_name}失败，报错信息为: {e}")
            else:
                response_text = await response.text()
                raise RuntimeError(f"解析文件{file_name}失败，报错信息为: {response_text}")
    
async def doc2docs(kb_name: str, file_name: str) -> dict:
    file_path = get_file_path(kb_name=kb_name, doc_name=file_name)
    from docx.table import _Cell, Table
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.text.paragraph import Paragraph
    from docx import Document

    doc = Document(file_path)
    text = ''

    def iter_block_items(parent):
        from docx.document import Document
        if isinstance(parent, Document):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            raise ValueError("DocParser parse fail")

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    for i, block in enumerate(iter_block_items(doc)):
        if isinstance(block, Paragraph):
            text += block.text.strip() + "\n"
        elif isinstance(block, Table):
            for row in block.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        text += paragraph.text.strip() + "\n"

    return {"file_name": file_name, "text": text}

async def markdown2docs(kb_name: str, file_name: str) -> dict:
    file_path = get_file_path(kb_name=kb_name, doc_name=file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return {"file_name": file_name, "text": text}

async def parser_docs(kb_name: str, file_name: str):
    if file_name.endswith('.pdf'):
        return await pdf2docs(kb_name, file_name)
    if file_name.endswith('.docx') or file_name.endswith('.doc'):
        return await doc2docs(kb_name, file_name)
    if file_name.endswith('.md') or file_name.endswith('.txt'):
        return await markdown2docs(kb_name, file_name)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    keep_separator=True,
    is_separator_regex=True,
    separators=["(?<=[。！？])"],
)

async def process_files(files: List[UploadFile], kb_name: str):

    ocr_text = {}
    milvus_rows = []
    params = [{"file_name": file.filename, "kb_name": kb_name} for file in files]
    # for result in run_in_thread_pool(parser_docs, params):
    # 使用 asyncio.gather 并发执行异步任务
    results = await asyncio.gather(*[parser_docs(kb_name, param["file_name"]) for param in params])

    for result in results:
        ocr_text[result["file_name"]] = []
        chunks = text_splitter.split_text(result["text"])
        add_docs_to_db(
            kb_name=kb_name, 
            file_name=result["file_name"], 
            file_ext=result["file_name"].split('.')[-1],
        )

        for i, chunk in enumerate(chunks):
            '''写入到milvus和sqlite中'''
            ocr_text[result["file_name"]].append({"id": i, "chunk": chunk})
            chunk_uuid = str(uuid.uuid4())
            add_summary_to_db(
                kb_name=kb_name,
                file_name=result["file_name"],
                chunk_id=i,
                chunk=chunk,
                chunk_uuid=chunk_uuid
            )
            vector = embedding_function([chunk])[0]

            milvus_rows.append(
                {
                    "id": chunk_uuid,
                    "kb_name": kb_name,
                    "file_name": result["file_name"],
                    "chunk": chunk,
                    "vector": vector,
                    "metadata": {},
                }
            )
        milvus_client.insert('langchat', milvus_rows)

