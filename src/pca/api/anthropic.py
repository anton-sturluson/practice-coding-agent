import os
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage, TextBlock

from pca.api.base import BaseLLMClient, LLMResponse, Message


class AnthropicClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        max_tokens: int = 65_536,
    ) -> None:
        self.model: str = model
        self.max_tokens: int = max_tokens
        self.client: AsyncAnthropic = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    async def call(
        self,
        messages: list[Message],
        enable_caching: bool = True,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse:
        system_messages: list[Message] = [m for m in messages if m.role == "system"]
        conversation_messages: list[Message] = [
            m for m in messages if m.role != "system"
        ]

        system_content: list[dict[str, Any]] | None = None
        if system_messages:
            system_content = []
            for i, msg in enumerate(system_messages):
                block: dict[str, Any] = {
                    "type": "text",
                    "text": msg.content,
                }
                if enable_caching and i == len(system_messages) - 1:
                    block["cache_control"] = {"type": "ephemeral"}
                system_content.append(block)

        messages: list[dict[str, Any]] = []
        for i, msg in enumerate(conversation_messages):
            is_last_message = i == len(conversation_messages) - 1

            content_blocks: list[dict[str, Any]] = [
                {"type": "text", "text": msg.content}
            ]

            if is_last_message and enable_caching:
                content_blocks[-1]["cache_control"] = {"type": "ephemeral"}

            messages.append({
                "role": msg.role,
                "content": content_blocks
            })

        response: AnthropicMessage = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_content,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )

        content: str = ""
        for block in response.content:
            if isinstance(block, TextBlock):
                content += block.text

        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_tokens=getattr(
                response.usage, "cache_creation_input_tokens", 0
            ),
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
        )
