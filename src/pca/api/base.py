from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class LLMResponse(BaseModel):
    content: str = Field(..., description="Generated content from the LLM")
    input_tokens: int = Field(default=0, description="Number of input tokens used")
    output_tokens: int = Field(default=0, description="Number of output tokens generated")
    cache_creation_tokens: int = Field(default=0, description="Tokens used to create cache")
    cache_read_tokens: int = Field(default=0, description="Tokens read from cache")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens


class BaseLLMClient(ABC):
    @abstractmethod
    async def call(self, messages: list[Message], **kwargs) -> LLMResponse:
        pass
