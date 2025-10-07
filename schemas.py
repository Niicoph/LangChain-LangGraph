from typing import List

from pydantic import BaseModel, Field

# This class is used to store the sources used by the agent


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The url of the source")


# This class is used to store the agent's response with answer and sources


class AgentResponse(BaseModel):
    """Schema for agent response with answer and resources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )
