from typing import Any
import os
import httpx
import asyncio
from datetime import datetime, date

from strands import Agent, tool
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp_client.client import get_streamable_http_mcp_client

app = BedrockAgentCoreApp()
log = app.logger

# Define a Streamable HTTP MCP Client
mcp_clients = [get_streamable_http_mcp_client()]

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant. Use tools when appropriate.
"""

# --- YOUR INTEGRATED TOOLS ---
@tool
def get_time() -> str:
    """Returns the current local time."""
    return datetime.now().strftime('%H:%M:%S')

@tool
def get_date() -> str:
    """Returns the current local date."""
    return str(date.today())

@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid calculation"

tools = [get_time, get_date, calculate]
_INLINE_FUNCTION_NAMES = set()

for mcp_client in mcp_clients:
    if mcp_client:
        tools.append(mcp_client)

def _make_conversation_manager():
    return NullConversationManager()

# --- CUSTOM OPENROUTER ADAPTER CLASS ---
class OpenRouterAdapter:
    def __init__(self):
        self.stateful = False
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "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")
        self.model_id = "openrouter/free"
        
    async def stream(self, messages, *args, **kwargs):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 1. TRANSLATE BEDROCK SCHEMA TO OPENROUTER SCHEMA
        or_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content_string = ""
            
            # Bedrock sends nested lists: [{"text": "Hello"}]
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and "text" in block:
                        content_string += block["text"]
            # Fallback if it's already a string
            elif isinstance(msg.get("content"), str):
                content_string = msg["content"]
                
            if content_string:
                or_messages.append({"role": role, "content": content_string})
        
        payload = {
            "model": self.model_id,
            "messages": or_messages,
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            # 2. ENHANCED ERROR LOGGING
            # If OpenRouter rejects it, print the exact reason to the console before crashing
            if response.status_code != 200:
                print(f"\n--- OPENROUTER API ERROR ---\n{response.text}\n----------------------------\n")
                
            response.raise_for_status()
            data = response.json()
            
        text_response = data["choices"][0]["message"].get("content", "")
        
        # 3. TRANSLATE OPENROUTER RESPONSE BACK TO BEDROCK STREAMING SCHEMA
        yield {
            "contentBlockStart": {
                "start": {"text": ""},
                "contentBlockIndex": 0
            }
        }
        yield {
            "contentBlockDelta": {
                "delta": {"text": text_response},
                "contentBlockIndex": 0
            }
        }
        yield {
            "messageStop": {
                "stopReason": "end_turn"
            }
        }
# ---------------------------------------

_agent = None

def get_or_create_agent():
    global _agent
    if _agent is None:
        _agent = Agent(
            # Pass the INSTANTIATED class, not a raw function
            model=OpenRouterAdapter(), 
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            tools=tools,
            conversation_manager=_make_conversation_manager(),
            hooks=[],
        )
    return _agent

# --- HARNESS UTILS ---
def _extract_prompt(payload: dict):
    if "messages" in payload:
        return payload["messages"]
    if "tool_results" in payload:
        return [{"role": "user", "content": [{"toolResult": {
            "toolUseId": tr["toolUseId"],
            "status": tr.get("status", "success"),
            "content": tr.get("content", []),
        }} for tr in payload["tool_results"]]}]
    return payload.get("prompt", "")

@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")
    agent = get_or_create_agent()
    prompt = _extract_prompt(payload)

    async for event in agent.stream_async(prompt):
        if not isinstance(event, dict) or "event" not in event:
            continue
        cbs = event["event"].get("contentBlockStart")
        if cbs is not None and not cbs.get("start"):
            continue
        yield event

if __name__ == "__main__":
    app.run()