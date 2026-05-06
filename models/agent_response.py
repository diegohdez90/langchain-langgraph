from typing import List
from pydantic import BaseModel, Field
from .source import Source


class AgentResponse(BaseModel):
    asnwer:str = Field(description="The agent's answer")
    sources:List[Source] = Field(
        default_factory=list,
        description="List of resources used to generate the answer")
