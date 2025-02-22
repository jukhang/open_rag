import os
from typing import List, Dict
from anyio import Semaphore
from typing import TypeVar, Callable
from typing_extensions import ParamSpec
from starlette.concurrency import run_in_threadpool
from typing import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_CONCURRENT_THREADS = 10
MAX_THREADS_GUARD = Semaphore(MAX_CONCURRENT_THREADS)

from settings import KB_ROOT_PATH

T = TypeVar("T")
P = ParamSpec("P")

async def run_async(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    async with MAX_THREADS_GUARD:
        return await run_in_threadpool(func, *args, **kwargs)


def validate_kb_name(kb_name: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in kb_name or kb_name.startswith("/"):
        return False
    return True

def get_kb_path(kb_name: str):
    return os.path.join(KB_ROOT_PATH, kb_name)

def get_doc_path(kb_name: str):
    return os.path.join(get_kb_path(kb_name))

def get_file_path(kb_name: str, doc_name: str) -> str:
    return os.path.join(get_doc_path(kb_name), doc_name)

def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    '''
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    '''
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            thread = pool.submit(func, **kwargs)
            tasks.append(thread)

        for obj in as_completed(tasks):
            yield obj.result()

