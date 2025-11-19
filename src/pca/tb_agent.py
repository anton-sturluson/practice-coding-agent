import asyncio
import logging
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent, FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from pca.main import AgenticWorkflow, WorkflowOutput
from pca.api.base import BaseLLMClient
from pca.api.anthropic import AnthropicClient
from pca.utils.logger import setup_logger


class TbAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "my-test-agent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        log_file = logging_dir / "run.log" if logging_dir else None
        logger = setup_logger("pca.agent", log_file=log_file)

        logger.info(f"Starting task:\n{instruction}")

        client: BaseLLMClient = AnthropicClient(model="claude-haiku-4-5")
        agent: AgenticWorkflow = AgenticWorkflow(client=client, logger=logger)

        session.start()
        output: WorkflowOutput = asyncio.run(agent.run(instruction, session))
        session.stop()

        logger.info(f"Task completed in {output.steps} steps")
        logger.info(
            f"Token usage - Input: {output.total_input_tokens}, Output: {output.total_output_tokens}"
        )

        output.print()

        return AgentResult(
            total_input_tokens=output.total_input_tokens,
            total_output_tokens=output.total_output_tokens,
            failure_mode=FailureMode.NONE,
        )
