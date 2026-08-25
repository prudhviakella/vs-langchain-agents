"""
2.1_mcp_server.py — A Local MCP Server
=======================================

This file is a self-contained MCP (Model Context Protocol) server.
It is NOT called directly in your notebook — LangChain launches it
as a subprocess when you configure it with transport="stdio".

Lifecycle:
  1. Notebook runs MultiServerMCPClient({"local_server": {"command": "python", "args": ["...mcp_server.py"]}})
  2. LangChain spawns:  python 2.1_mcp_server.py
  3. This process starts and blocks on mcp.run(transport="stdio")
  4. The MCP protocol handshake happens over stdin/stdout
  5. LangChain discovers the tools/resources/prompts defined below
  6. When the notebook kernel stops, this subprocess is killed

What this server exposes:
  - Tool:     search_web(query)       — Tavily web search
  - Resource: github://...README.md   — the langchain-mcp-adapters README
  - Prompt:   prompt()                — a LangChain expert system prompt

FastMCP is the server framework (from the `mcp` package). It handles
the protocol details — you just write Python functions and decorate them.
"""

# ============================================================
# Setup: Load Keys and Import Dependencies
# ============================================================
# load_dotenv() is called here because this file runs as its own
# Python process — it doesn't inherit the parent notebook's environment.
# Each subprocess needs to load its own .env.

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from typing import Dict, Any
from requests import get


# ============================================================
# Initialise the MCP Server
# ============================================================
# FastMCP("mcp_server") creates the server object.
# The string "mcp_server" is the server's display name —
# it appears in logs and error messages. It doesn't affect routing.

mcp = FastMCP("mcp_server")


# Tavily client for web search — reads TAVILY_API_KEY from env
tavily_client = TavilyClient()


# ============================================================
# TOOL: search_web
# ============================================================
# @mcp.tool() is FastMCP's equivalent of LangChain's @tool.
# It registers this function as a callable tool on the server.
#
# When the notebook calls client.get_tools(), this function's:
#   name        → "search_web"  (from the function name)
#   description → "Search the web for information"  (from the docstring)
#   schema      → {query: str}  (from the type hints)
# are sent to LangChain, which wraps it as a standard tool object.
#
# The agent then calls this tool exactly like a local @tool —
# the MCP layer handles serialisation and subprocess communication
# transparently.

@mcp.tool()
def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    results = tavily_client.search(query)
    return results


# ============================================================
# RESOURCE: github_file
# ============================================================
# @mcp.resource() exposes read-only data that agents can access.
# The URI string ("github://...") is the resource's address —
# agents use this URI to request the resource.
#
# Resources are like files or documents: the agent reads them
# for context but cannot modify them. They complement tools:
#   Tools    → DO something (search, calculate, send)
#   Resources → READ something (a doc, a file, a config)
#
# In this case, the resource fetches the README.md of the
# langchain-mcp-adapters repository from GitHub raw content.
# This gives the agent access to up-to-date library documentation
# without embedding it in the system prompt.
#
# Note: the URL in the implementation uses 'blob' which may need
# to be 'refs/heads' for the GitHub raw API — check if you get 404s.

@mcp.resource("https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/blob/main/README.md")
def github_file():
    """
    Resource for accessing langchain-ai/langchain-mcp-adapters/README.md file
    """
    url = "https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/blob/main/README.md"
    try:
        resp = get(url)
        return resp.text
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# PROMPT: prompt
# ============================================================
# @mcp.prompt() defines a named system prompt template.
# The notebook fetches it with:
#   client.get_prompt("local_server", "prompt")
# and uses it as the agent's system_prompt.
#
# This is powerful because:
#   1. The server author controls the agent's persona and constraints
#   2. Consumers don't need to write system prompts for domains they
#      don't fully understand (the server expert wrote it for them)
#   3. The prompt can be updated server-side without touching consumer code
#
# Notice the key design decisions in this prompt:
#   - Scopes the agent to LangChain/LangGraph/LangSmith only
#   - Explicitly lists available tools so the agent knows to use them
#   - Gives a graceful refusal message for off-topic questions
#   - Allows multiple tool calls ("You may try multiple...")
#   - Allows clarifying questions ("You may also ask...")

@mcp.prompt()
def prompt():
    """Analyze data from a langchain-ai repo file with comprehensive insights"""
    return """
    You are a helpful assistant that answers user questions about LangChain, LangGraph and LangSmith.

    You can use the following tools/resources to answer user questions:
    - search_web: Search the web for information
    - github_file: Access the langchain-ai repo files

    If the user asks a question that is not related to LangChain, LangGraph or LangSmith, you should say "I'm sorry, I can only answer questions about LangChain, LangGraph and LangSmith."

    You may try multiple tool and resource calls to answer the user's question.

    You may also ask clarifying questions to the user to better understand their question.
    """


# ============================================================
# Entry Point: Start the Server
# ============================================================
# mcp.run(transport="stdio") starts the event loop and begins
# listening for MCP protocol messages on stdin, responding on stdout.
#
# The if __name__ == "__main__" guard ensures this only runs when
# the file is executed directly (python 2.1_mcp_server.py),
# NOT when it is imported by another module.
#
# transport="stdio" is correct for local subprocess servers.
# For an HTTP server you would use transport="http" instead.

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
