"""
3.5_email_agent.py — Production Email Agent Script
===================================================

This is the deployable version of the email agent from notebook 3.5.
It combines all Module 3 advanced features into one cohesive application:

  1. Custom state    → AuthenticatedState tracks login status across turns
  2. Runtime context → EmailContext injects credentials server-side
  3. Tool gating     → dynamic_tool_call restricts tools by auth status
  4. Dynamic prompt  → dynamic_prompt_func changes persona by auth status
  5. HITL            → HumanInTheLoopMiddleware pauses before send_email

Key difference from the notebook:
  - dynamic_tool_call is declared `async` here (await handler(request))
  - This is required in production async environments (FastAPI, etc.)
  - The notebook uses synchronous mode; the .py file uses async
  - No InMemorySaver — the deployment platform injects its own checkpointer
  - The `agent` object is exported at module level for langgraph.json

Security design:
  - Credentials never appear in conversation history (injected via context)
  - Unauthenticated sessions cannot call email tools (dynamic tool gating)
  - Every send_email call requires explicit human approval (HITL)
"""

# ============================================================
# Imports and Environment Setup
# ============================================================
# load_dotenv() must be first — libraries read env vars at import time.

from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain.agents.middleware import (
    wrap_model_call, 
    dynamic_prompt, 
    HumanInTheLoopMiddleware,
    ModelRequest, 
    ModelResponse
)
from typing import Callable

load_dotenv()


# ============================================================
# Context Schema — Server-Side User Credentials
# ============================================================
# EmailContext carries credentials injected per request by the server.
# These NEVER come from the user's message — they come from your
# auth session store. The agent reads them via runtime.context
# inside the authenticate tool.
#
# Default values are for demo only — production would have no defaults
# and would always require the caller to supply real credentials.

@dataclass
class EmailContext:
    email_address: str = "julie@example.com"  # Server-side credential
    password: str = "password123"              # Server-side credential


# ============================================================
# Custom State Schema — Authentication Status
# ============================================================
# AuthenticatedState persists the authentication result across turns.
# Once set to True by the authenticate tool, it stays True for the
# lifetime of the thread (until the checkpointer is cleared).
#
# This field drives both dynamic_tool_call and dynamic_prompt_func —
# they both read request.state.get("authenticated") to make decisions.

class AuthenticatedState(AgentState):
    authenticated: bool   # Written by authenticate tool; read by middleware


# ============================================================
# Tool 1: check_inbox — Read-Only, Auto-Execute
# ============================================================
# Returns a hardcoded fake inbox entry for demonstration.
# In production: call Gmail/Outlook/IMAP API here.
# No ToolRuntime needed — this tool has no side effects and
# doesn't need to read state or context.

@tool
def check_inbox() -> str:
    """Check the inbox for recent emails"""
    # Production: fetch from email API using credentials from context
    return """
    Hi Julie, 
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """


# ============================================================
# Tool 2: send_email — High-Stakes, HITL-Protected
# ============================================================
# Simulates sending an email. HITL middleware will pause BEFORE
# this tool executes, requiring human approval.
# In production: call Gmail API / SendGrid / SES here.

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an response email"""
    # Production: call email sending API
    return f"Email sent to {to} with subject {subject} and body {body}"


# ============================================================
# Tool 3: authenticate — Login, Writes to State
# ============================================================
# The authenticate tool:
#   - Takes email and password arguments (extracted from user's message by LLM)
#   - Compares them against runtime.context (the real credentials, server-side)
#   - Returns a Command that writes authenticated=True/False to state
#   - Includes a ToolMessage manually (required when returning Command)
#
# Security note: the comparison is intentionally simple for this demo.
# Production would use bcrypt/argon2 for password comparison and
# would not store plaintext passwords anywhere.

@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the user with the given email and password"""
    if email == runtime.context.email_address and password == runtime.context.password:
        # Credentials match — grant access
        return Command(
            update={
                "authenticated": True,  # Unlocks email tools in next turn
                "messages": [
                    ToolMessage("Successfully authenticated", tool_call_id=runtime.tool_call_id)
                ],
            }
        )
    else:
        # Credentials don't match — deny access
        return Command(
            update={
                "authenticated": False,  # Keeps email tools locked
                "messages": [
                    ToolMessage("Authentication failed", tool_call_id=runtime.tool_call_id)
                ],
            }
        )


# ============================================================
# Middleware 1: Dynamic Tool Selection (Auth-Gated)
# ============================================================
# Runs before every LLM call. Reads authentication status from state
# and presents a completely different tool set:
#
#   authenticated=False (or None):  tools = [authenticate]
#   authenticated=True:            tools = [check_inbox, send_email]
#
# This is declared `async` to work in async deployment environments.
# The notebook version is sync — this version uses `await handler(request)`.
#
# The handler must be awaited in async context. Forgetting await here
# causes the LLM call to never execute.

@wrap_model_call
async def dynamic_tool_call(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Allow inbox/send email tools only after successful authentication"""

    authenticated = request.state.get("authenticated")  # None on first call

    if authenticated:
        tools = [check_inbox, send_email]   # Email capabilities unlocked
    else:
        tools = [authenticate]              # Only login allowed pre-auth

    request = request.override(tools=tools)
    return await handler(request)  # async — must await in production deployment


# ============================================================
# Middleware 2: Dynamic System Prompt (Auth-Aware Persona)
# ============================================================
# Generates the system prompt fresh on every LLM call.
# The persona changes based on whether the user has authenticated.
#
# Note: the function name 'dynamic_prompt_func' avoids shadowing
# the imported 'dynamic_prompt' decorator (the name collision bug
# that could occur if both were named 'dynamic_prompt').

authenticated_prompt = "You are a helpful assistant that can check the inbox and send emails."
unauthenticated_prompt = "You are a helpful assistant that can authenticate users."


@dynamic_prompt
def dynamic_prompt_func(request: ModelRequest) -> str:
    """Generate system prompt based on authentication status"""
    authenticated = request.state.get("authenticated")

    if authenticated:
        return authenticated_prompt   # Email assistant persona
    else:
        return unauthenticated_prompt  # Login assistant persona


# ============================================================
# Agent Assembly — The Exported Object
# ============================================================
# This agent object is what langgraph.json references:
#   "agent": "./3.5_email_agent.py:agent"
#
# Middleware order is significant:
#   1. dynamic_tool_call   → must run first to set the correct tool list
#   2. dynamic_prompt_func → uses auth state to set the prompt
#   3. HumanInTheLoopMiddleware → needs the tool list to know which to watch
#
# No checkpointer here — injected by the deployment platform.
# If running this locally without a platform, add checkpointer=InMemorySaver().
#
# tools=[authenticate, check_inbox, send_email] is the MAXIMUM set.
# dynamic_tool_call reduces this to the appropriate subset per request.

agent = create_agent(
    "gpt-5-nano",
    tools=[authenticate, check_inbox, send_email],  # Maximum possible tool set
    state_schema=AuthenticatedState,    # Custom state with authenticated field
    context_schema=EmailContext,        # Server-side credential injection
    middleware=[
        dynamic_tool_call,              # Auth-gated tool filtering
        dynamic_prompt_func,            # Auth-aware persona
        HumanInTheLoopMiddleware(
            interrupt_on={
                "authenticate": False,  # Auto-execute — login is always safe
                "check_inbox":  False,  # Auto-execute — reads are safe
                "send_email":   True,   # PAUSE — irreversible side effect
            }
        ),
    ],
)
