from harbor import BaseAgent, BaseEnvironment, AgentContext

class TestAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        """The name of the agent."""
        return "test-agent"

    def version(self) -> str | None:
        """The version of the agent."""
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Run commands to setup the agent & its tools.
        """
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Runs the agent in the environment. Be sure to populate the context with the
        results of the agent execution. Ideally, populate the context as the agent
        executes in case of a timeout or other error.

        Args:
            instruction: The task instruction.
            environment: The environment in which to complete the task.
            context: The context to populate with the results of the agent execution.
        """
        print(f"Instruction: {instruction}")
        print(f"Environment: {environment}")
        print(f"Context: {context}")

        # result = await environment.exec("echo 'Hello from agent!' && touch /app/hello.txt && echo 'Hello, World!' > /app/hello.txt")
        result = await environment.exec("ech")
        print(f"Command result: {result}")
        print(dir(result))
        # self._logger.info(f"Command result: {result}")

        # context.record_action("RECORD: Hello, World!")
