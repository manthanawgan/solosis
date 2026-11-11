from pydantic import BaseModel, Field
from typing import Literal


class Memory(BaseModel):
    content: str = Field(
        description = "one atomic, independently useful fact about the user"
    )

    category: Literal[
        "profile",
        "preference",
        "goal",
        "project",
        "skill",
        "constraint",
        "episodic"
    ]

    importance: int = Field(
        ge = 1,
        le = 10,
        description = "Preliminary importance score from 1 to 10"            
    )

    confidence: int = Field(
        ge = 0.0,
        le = 1.0,
        description = "How confidently the source supports this memory"
    )

    source_message_ids: list[str] = Field(
        description = "IDs of messages supporting this memory"
    )


    class MemoryExtraction(BaseModel):
        memories: list[Memory]