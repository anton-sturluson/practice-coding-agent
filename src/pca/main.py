"""
Main Agent class that can be exported to TB or Harbor.

Inspired by: https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py
"""

import logging
import re

from pydantic import BaseModel
from terminal_bench.terminal.tmux_session import TmuxSession

from pca.api.base import BaseLLMClient, Message, LLMResponse


class WorkflowOutput(BaseModel):
    total_input_tokens: int
    total_output_tokens: int
    steps: int
    messages: list[Message]

    def print(self):
        print(f"Total input tokens: {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        print(f"Steps taken: {self.steps}")
        print("Messages:")
        for msg in self.messages:
            print(f"\t[{msg.role}]: {msg.content.strip()}")


class WorkflowConfig(BaseModel):
    system_prompt: str = """
    You're a helpful coding agent. You will be given a task which you must complete.
    Start by planning what needs to be done to complete the task.
    At the end of your plan, output one bash command to execute at a time.
    The command must be enclosed in triple backticks bash block.

    If you have completed the task, output the following command in a triple
    backticks bash block: COMPLETE_TASK
    """
    instruction_prompt: str = "Task:\n{instruction}"

class AgenticWorkflow:
    def __init__(
        self,
        client: BaseLLMClient,
        config: WorkflowConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.messages: list[Message] = []
        self.steps: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.client: BaseLLMClient = client
        self.config: WorkflowConfig = config or WorkflowConfig()
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    async def run(
        self,
        instruction: str,
        session: TmuxSession,
    ) -> str:
        self.logger.info("Starting workflow")

        self.messages: list[Message] = [
            Message(role="system", content=self.config.system_prompt),
            Message(
                role="user",
                content=self.config.instruction_prompt.format(instruction=instruction)
            ),
        ]

        while True:
            halting_condition: bool = await self.step(session)
            self.steps += 1
            if halting_condition:
                self.logger.info("Task marked as complete")
                break

        self.logger.info(f"Workflow completed in {self.steps} steps")

        return WorkflowOutput(
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            steps=self.steps,
            messages=self.messages,
        )

    async def step(self, session: TmuxSession) -> bool:
        self.logger.info(f"\n*************Step {self.steps + 1}*************")
        response: LLMResponse = await self.client.call(self.messages)
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens

        self.logger.info(f"LLM Response: {response.content[:200]}...")

        self.messages.append(Message(role="assistant", content=response.content))
        action: str = await self.parse_action(response.content)

        if await self.has_finished(action):
            return True

        output: str | None = None
        if action:
            self.logger.info(f"Executing: {action}")
            output: str = await self.execute_action(session, action)
            self.logger.info(f"Output: {output[:200]}...")
            self.messages.append(Message(role="user", content=f"Output: {output}"))
        else:
            self.logger.warning("Failed to parse bash command from LLM response")
            self.messages.append(
                Message(
                    role="user",
                    content=f"Prasing error: {response.content}. Bash command must "
                    "be enclosed in triple backticks bash block."
                )
            )

        return False

    async def parse_action(self, response: str) -> str:
        """Parse bash command from LLM response.

        Returns:
            Command string if found, None if COMPLETE_TASK signal detected.
        """
        bash_pattern = r"```bash\n(.*?)\n```"
        match = re.search(bash_pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        return ""

    async def execute_action(self, session: TmuxSession, action: str) -> str:
        session.send_keys([action, "Enter"])
        output: str = session.get_incremental_output().strip()
        return output

    async def has_finished(self, action: str) -> bool:
        return "COMPLETE_TASK" in action
