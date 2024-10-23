from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from timescale_vector import client


Metadata = Dict[str, Any]
EmbeddingVector = List[float]
TimeRange = Tuple[datetime, datetime]


class SearchResult(TypedDict):
    id: str
    content: str
    metadata: Metadata
    distance: float
    search_type: str = "semantic"


class SearchOptions(TypedDict, total=False):
    limit: int = 5
    predicates: Optional[client.Predicates] = None
    metadata_filter: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    time_range: Optional[TimeRange] = None
    return_dataframe: bool = True
