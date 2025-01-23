import nest_asyncio
nest_asyncio.apply()

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from rich.console import Console
from rich.panel import Panel
import configparser
import os

console = Console()

# # ollama本地部署
def get_model_client_ollama() -> OpenAIChatCompletionClient:  # type: ignore
    return OpenAIChatCompletionClient(
        model="llama3.3-extra:latest",
        api_key="ollama",
        base_url="http://ollama.webtw.xyz:11434/v1",
        model_capabilities={
            "json_output": False,
            "vision": True,
            "function_calling": True,
        },
    )

def load_config():
    config = configparser.ConfigParser()
    
    # 檢查設定檔是否存在
    if not os.path.exists('config.ini'):
        raise FileNotFoundError('請建立 config.ini 檔案（可以從 config.example.ini 複製）')
    
    config.read('config.ini')
    return config

# 讀取設定
config = load_config()
API_KEY = config['API']['api_key_mistral']

async def main() -> None:

    model_client = get_model_client_ollama()

    agent1 = AssistantAgent("Assistant1", model_client=model_client)
    agent2 = AssistantAgent("Assistant2", model_client=model_client)

    termination = MaxMessageTermination(11)

    team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)

    stream = team.run_stream(task="Count from 1 to 10, respond one at a time.")
    
    async for message in stream:
        if hasattr(message, 'content'):
            # Print message in panel with source as title
            console.print(Panel(
                message.content,
                title=f"[bold blue]{message.source}[/bold blue]",
                border_style="blue"
            ))
            
            # Print usage statistics if available
            if message.models_usage:
                console.print(f"[dim]Usage - Prompt tokens: {message.models_usage.prompt_tokens}, "
                         f"Completion tokens: {message.models_usage.completion_tokens}[/dim]")
            console.print("―" * 80)  # Separator line
        else:  # TaskResult object
            console.print("\n[bold yellow]Task Result Summary[/bold yellow]")
            console.print(f"Stop Reason: {message.stop_reason}")
            console.print("―" * 80)

asyncio.run(main())
