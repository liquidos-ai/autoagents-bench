import asyncio
from _src.langgraph_bench import run_langgraph
from _src.crewai_bench import run_crewai


async def main() -> None:
    await run_langgraph()
    await run_crewai()


if __name__ == "__main__":
    asyncio.run(main())
