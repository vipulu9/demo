from typing import Any
import os
import httpx
import asyncio
import boto3
import json
from datetime import datetime, date

from strands import Agent, tool
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp_client.client import get_streamable_http_mcp_client

app = BedrockAgentCoreApp()
log = app.logger

# Define a Streamable HTTP MCP Client
mcp_clients = [get_streamable_http_mcp_client()]

# Fallback prompt in case the Bedrock API fails
FALLBACK_SYSTEM_PROMPT = """
You are a helpful assistant. Use tools when appropriate.
"""

# --- BEDROCK PROMPT MANAGEMENT ---
def fetch_bedrock_prompt(user_name: str = "Guest") -> str:
    """
    Fetches the prompt dynamically from AWS Bedrock and injects variables.
    """
    prompt_id = os.environ.get("BEDROCK_PROMPT_ID", "arn:aws:bedrock:us-east-1:807598718319:prompt/5I3LESIERR")
    
    if prompt_id == "YOUR_PROMPT_ID_HERE":
        log.warning("⚠️ No BEDROCK_PROMPT_ID found. Using fallback prompt.")
        return FALLBACK_SYSTEM_PROMPT
        
    try:
        client = boto3.client('bedrock-agent', region_name='us-east-1')
        response = client.get_prompt(
            promptIdentifier=prompt_id,
            promptVersion='DRAFT' 
        )
        
        # THE FIX: Boto3 puts 'variants' right at the root level!
        variants = response.get('variants', [])
        
        raw_text = FALLBACK_SYSTEM_PROMPT
        if variants:
            config = variants[0].get('templateConfiguration', {})
            
            # Deep search for the text in standard text format
            if 'text' in config:
                raw_text = config['text'].get('text', FALLBACK_SYSTEM_PROMPT)
            
            # Deep search for the text in chat format (system OR user message)
            elif 'chat' in config:
                if 'system' in config['chat'] and config['chat']['system']:
                    raw_text = config['chat']['system'][0].get('text', FALLBACK_SYSTEM_PROMPT)
                elif 'messages' in config['chat'] and config['chat']['messages']:
                    raw_text = config['chat']['messages'][0]['content'][0].get('text', FALLBACK_SYSTEM_PROMPT)
        
        # 1. INJECT THE VARIABLE LOCAL LAYER
        final_prompt = raw_text.replace("{{user_name}}", user_name)
        
        # 2. VERIFICATION LOGGING
        log.info(f"✅ SUCCESS: Loaded prompt from AWS Bedrock! Prompt text: {final_prompt}")
        
        return final_prompt

    except Exception as e:
        log.error(f"❌ Failed to fetch prompt from Bedrock: {e}. Using fallback.")
        return FALLBACK_SYSTEM_PROMPT
# ---------------------------------

# --- CONFIGURATION LAYER ---
def create_agent(user_name: str = "Guest"):
    """
    Creates a fresh agent on every request to ensure the latest 
    Bedrock prompt is loaded without requiring a container restart.
    """
    return Agent(
        model=OpenRouterAdapter(), 
        system_prompt=fetch_bedrock_prompt(user_name=user_name),
        tools=tools,
        conversation_manager=_make_conversation_manager(),
        hooks=[],
    )

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
            
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and "text" in block:
                        content_string += block["text"]
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
    
    incoming_user = "Guest"
    raw_text = payload.get("prompt", "")
    
    # 1. THE TRICK: Check if the text from the chat box is actually a JSON string
    try:
        if isinstance(raw_text, str) and raw_text.strip().startswith("{"):
            parsed_data = json.loads(raw_text)
            
            # Extract the username from the parsed string
            incoming_user = parsed_data.get("user_name", "Guest")
            
            # Fix the payload so the agent only sees the prompt and not the raw JSON
            payload["prompt"] = parsed_data.get("prompt", "")
    except json.JSONDecodeError:
        pass # It was just a normal chat message, do nothing
        
    # 2. Fallback: If it wasn't a JSON string, check the root payload 
    if incoming_user == "Guest":
        incoming_user = payload.get("user_name", "Guest")
    
    # 3. Inject it into the agent
    agent = create_agent(user_name=incoming_user) 
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