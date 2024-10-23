from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union


from timescale_vector import client


Metadata = Dict[str, Any]
EmbeddingVector = List[float]
TimeRange = Tuple[datetime, datetime]


class SearchResult(TypedDict, total=False):
    id: str
    content: str
    metadata: dict
    distance: float
    search_type: Literal["semantic", "fulltext"]


class HybridSearchResult(TypedDict, total=False):
    id: str
    content: str
    metadata: dict
    semantic_distance: Optional[float]
    fulltext_distance: Optional[float]


class SearchOptions(TypedDict, total=False):
    limit: int
    predicates: Optional[client.Predicates]
    metadata_filter: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
    time_range: Optional[TimeRange]
    return_dataframe: bool
