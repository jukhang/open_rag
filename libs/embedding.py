#! python3
# -*- encoding: utf-8 -*-
'''
@File    : embedding.py
@Time    : 2024/09/05 13:15:59
@Author  : longfellow
@Version : 1.0
@Email   : longfellow.wang@gmail.com
'''



import numpy as np
from openai import Client
from tenacity import retry
from numpy.typing import NDArray
from typing import List, Dict, Any, Union
from typing import Union, List, TypeVar, Sequence
from typing import  cast, Protocol, runtime_checkable


# Documents
Document = str
Documents = List[Document]

# Images
ImageDType = Union[np.uint, np.int_, np.float64]  # type: ignore[name-defined]
Image = NDArray[ImageDType]
Images = List[Image]

# Embeddings
Vector = Union[Sequence[float], Sequence[int]]
Embedding = Vector
Embeddings = List[Embedding]

Embeddable = Union[Documents, Images]

T = TypeVar("T")
D = TypeVar("D", bound=Embeddable, contravariant=True)

# OneOrMany
OneOrMany = Union[T, List[T]]

def cast_embeddings(
    target: Union[OneOrMany[Embedding], OneOrMany[np.ndarray]]  # type: ignore[type-arg]
) -> Embeddings:
    '''Cast object type to Embeddings.'''
    if isinstance(target, List):
        if isinstance(target[0], (int, float)):
            return cast(Embeddings, [target])

    return cast(Embeddings, target)

def validate_embeddings(embeddings: Embeddings) -> Embeddings:
    """Validates embeddings to ensure it is a list of list of ints, or floats"""
    if not isinstance(embeddings, list):
        raise ValueError(
            f"Expected embeddings to be a list, got {type(embeddings).__name__}"
        )
    if len(embeddings) == 0:
        raise ValueError(
            f"Expected embeddings to be a list with at least one item, got {len(embeddings)} embeddings"
        )
    if not all([isinstance(e, list) for e in embeddings]):
        raise ValueError(
            "Expected each embedding in the embeddings to be a list, got "
            f"{list(set([type(e).__name__ for e in embeddings]))}"
        )
    for i, embedding in enumerate(embeddings):
        if len(embedding) == 0:
            raise ValueError(
                f"Expected each embedding in the embeddings to be a non-empty list, got empty embedding at pos {i}"
            )
        if not all(
            [
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in embedding
            ]
        ):
            raise ValueError(
                "Expected each value in the embedding to be a int or float, got an embedding with "
                f"{list(set([type(value).__name__ for value in embedding]))} - {embedding}"
            )
    return embeddings

@runtime_checkable
class EmbeddingFunction(Protocol[D]):
    def __call__(self, input: D) -> Embeddings:
        ...

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # Raise an exception if __call__ is not defined since it is expected to be defined
        call = getattr(cls, "__call__")

        def __call__(self: EmbeddingFunction[D], input: D) -> Embeddings:
            result = call(self, input)
            return validate_embeddings(cast_embeddings(result))

        setattr(cls, "__call__", __call__)

    def embed_with_retries(
        self, input: D, **retry_kwargs: Dict[str, Any]
    ) -> Embeddings:
        return cast(Embeddings, retry(**retry_kwargs)(self.__call__)(input))



class OpenAIEmbeddingsFunction(EmbeddingFunction[Documents]):
    def __init__(self, api_key: str, base_url: str, model_name: str, **kwargs):
        try:
            self.kwargs = kwargs
            self.model_name = model_name
            self.client = Client(api_key=api_key, base_url=base_url)
        except Exception as e:
            raise RuntimeError(f"Error initializing OpenAI Client: {e}")

    def __call__(self, input: Documents) -> Embeddings:
        input = [t.replace("\n", " ") for t in input]
        try:
            response = self.client.embeddings.create(
                model = self.model_name,
                input = input,
            )
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI Embeddings API: {e}")
        
        return [data.embedding for data in response.data]


from settings import settings

embedding_function = OpenAIEmbeddingsFunction(
    api_key=settings.XINFERENCE_API_KEY, 
    base_url=f"http://{settings.XINFERENCE_HOST}:{settings.XINFERENCE_PORT}/v1", 
    model_name=settings.EMBEDDING_MODEL
)
