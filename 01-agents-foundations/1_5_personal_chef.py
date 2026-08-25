"""
1.5_personal_chef.py — Production-Ready Agent Script
=====================================================

This file is the deployable version of the Personal Chef agent.
Unlike the notebook (1.5_personal_chef.ipynb), which is for
exploration and learning, this .py file is designed to be:

  1. Imported by the LangGraph deployment platform (via langgraph.json)
  2. Run as a standalone script during development/testing
  3. Extended with additional tools or memory backends

The key difference from the notebook:
  - No InMemorySaver here — the deployment platform manages memory
    externally (e.g. via a database checkpointer it injects at runtime)
  - The `agent` object is exported at module level so langgraph.json
    can reference it as "1.5_personal_chef.py:agent"

How langgraph.json uses this file:
  {
    "graphs": {
      "agent": "./1.5_personal_chef.py:agent"  ← imports this module,
    }                                               reads `agent` variable
  }
"""

# ============================================================
# Step 1: Load Environment Variables
# ============================================================
# load_dotenv() must be called BEFORE any library that reads
# environment variables (LangChain, Tavily, OpenAI SDK, etc.).
# In this script it's called at the top of the file so it runs
# automatically whenever the module is imported.

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Step 2: Define the Web Search Tool
# ============================================================
# Tavily is the search backend. We wrap it in @tool so LangChain
# can pass its name, description, and schema to the LLM.
#
# The model reads the docstring ("Search the web for information")
# and decides autonomously when to call this tool.
#
# Dict[str, Any] as the return type reflects the Tavily response:
#   {
#     "results": [
#       {"title": "...", "url": "...", "content": "..."},
#       ...
#     ],
#     "answer": "..."   # optional Tavily-generated summary
#   }
# The model reads this dict and extracts the relevant recipe info.

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()  # Reads TAVILY_API_KEY from environment

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

# ============================================================
# Step 3: Define the System Prompt
# ============================================================
# The system prompt sets the agent's persona and operational rules.
# Three key instructions embedded here:
#
#   1. ROLE:   "You are a personal chef"
#              → establishes tone and domain expertise
#
#   2. INPUT:  "The user will give you a list of ingredients"
#              → tells the model what to expect from the user
#
#   3. METHOD: "Using the web search tool, search the web..."
#              → explicitly instructs tool use (prevents the model
#                from hallucinating recipes from training data)
#
#   4. OUTPUT: "Return recipe suggestions and eventually instructions"
#              → two-step response: suggest first, detail on request
#                This keeps first replies concise and conversational

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

# ============================================================
# Step 4: Create and Export the Agent
# ============================================================
# This `agent` variable is the module's main export.
# langgraph.json points to it as "1.5_personal_chef.py:agent".
#
# Notice: NO checkpointer here.
# In the notebook we used InMemorySaver for quick demos.
# In production, the LangGraph platform injects its own
# persistence backend (e.g. a PostgreSQL checkpointer) at
# deployment time — we don't hardcode it in the script.
#
# The platform also handles:
#   - thread_id management (one per user session)
#   - authentication and rate limiting
#   - horizontal scaling across multiple workers
#
# Our job is just to define the agent object correctly.

from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt=system_prompt
    # checkpointer is injected by the deployment platform at runtime
)
