from pydantic import BaseModel, Field

class Source(BaseModel):
    """Represents a source of information."""
    url:str = Field(
        description="URL of the resource"
    )
    description: str
