import ollama
import requests
import json
import time

from pydantic import BaseModel, ValidationError
from typing import Literal, Dict, Type, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from googlesearch import search
from bs4 import BeautifulSoup
from datetime import date




# Tool Base (
# ─────────────────────────────────────────────────────────────

class Tool(BaseModel):

    @classmethod
    def description(cls) -> str:
        return cls.__doc__ or f"{cls.__name__} tool"

    @classmethod
    def schema(cls) -> dict:
        return {
            "type": "function",
            "function": {
                "name": cls.__name__,
                "description": cls.description(),
                "parameters": cls.model_json_schema(),
            },
        }


# ─────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────

class CalculateTool(Tool):
    """ Simple arthemetic Calculator"""
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float
    
    def run(self):
        match self.operation:
            case "add": return self.a + self.b
            case "subtract": return self.a - self.b
            case "multiply": return self.a * self.b
            case "divide":
                if self.b == 0:
                    raise ValueError("Division by zero")
                return self.a / self.b


class WebSearchTool(Tool):
    """Get  urls for the query."""
    query: str
    
    def run(self):
        try:
            time.sleep(1)
            
            urls = list(search(self.query, num_results=5))
            
            return [{"url": u} for u in urls if u.startswith("http")] or [{"url": "https://news.google.com"}]
        
        except Exception:
            return [{"url": "https://news.google.com"}]


class FetchPageTool(Tool):
    """"Get the page data for the urls"""
    url: str | None = None
    urls: List[str] | None = None
    
    def run(self):
        urls = [self.url] if self.url else self.urls or []
        urls = [u for u in urls if u and u.startswith("http")]

        results = []

        for u in urls:
            try:
                res = requests.get(u, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                
                res.raise_for_status()

                soup = BeautifulSoup(res.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.extract()

                text = soup.get_text(separator=" ", strip=True)

                results.append({
                    "url": u,
                    "content": text[:2000],
                })

            except Exception:
                continue

        return results if results else [{"error": "Failed to fetch"}]


class SummarizeTool(Tool):
    """Summarise the content of the pages  """
    content: str | list[dict]
    
    def run(self):
        if isinstance(self.content, list):
            text = "\n\n".join(
                c.get("content", str(c)) if isinstance(c, dict) else str(c)
                for c in self.content
            )
        else:
            text = str(self.content)

        if not text.strip():
            return "Error: Nothing to summarize"

        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize in 5 concise bullet points."},
                {"role": "user", "content": text[:4000]},
            ],
            options={"temperature": 0.2},
        )

        return response.message.content.strip()


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────

TOOLS: Dict[str, Type[Tool]] = {
    t.__name__: t
    for t in [CalculateTool, WebSearchTool, FetchPageTool, SummarizeTool]
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def serialize_content(content):
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    
    except Exception:
        return str(content)


def is_valid_content(content):
    if not content:
        return False

    if isinstance(content, str):
        return not content.startswith("Error")

    if isinstance(content, list):
        return any("error" not in i for i in content if isinstance(i, dict))

    if isinstance(content, dict):
        return "error" not in content

    return True


# ─────────────────────────────────────────────────────────────
# Parallel Execution
# ─────────────────────────────────────────────────────────────

def execute_tool_call(call):
    name = call.function.name
    args = call.function.arguments
    tool_cls = TOOLS.get(name)

    if not name or tool_cls is None:
        return {"role": "tool", "name": name or "unknown", "content": {"error": "Unsupported tool"}}

    try:
        tool = tool_cls.model_validate(args)
        result = tool.run()

        return {
            "role": "tool",
            "name": name,
            "content": result,  # structured
        }

    except (ValidationError, ValueError) as e:
        return {"role": "tool", "name": name, "content": {"error": str(e)}}


def execute_parallel(tool_calls):
    with ThreadPoolExecutor(max_workers=min(8, len(tool_calls))) as executor:
        futures = [executor.submit(execute_tool_call, call) for call in tool_calls]
        return [f.result() for f in as_completed(futures)]


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

MODEL = "llama3.1:latest"
today = date.today()
SYSTEM = f"""
You are an intelligent assistant.

Today's date is {today}.

STRICT WORKFLOW:
1. Use WebSearchTool
2. Use FetchPageTool (parallel)
3. Use SummarizeTool

If fetch fails, summarize search results.

NEVER return empty response.
"""


def agent(user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_input},
    ]

    print(f"\nUser: {user_input}")

    while True:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=[t.schema() for t in TOOLS.values()],
        )

        msg = response.message
        messages.append(msg)

        if not msg.tool_calls:
            print(f"Agent: {msg.content}")
            return msg.content

        results = execute_parallel(msg.tool_calls)

        valid_results = []

        for res in results:
            if is_valid_content(res["content"]):
                print(f"[tool] {res['name']} → OK")
                valid_results.append(res)
            else:
                print(f"[tool] {res['name']} → skipped")

        if not valid_results:
            messages.append({
                "role": "system",
                "content": "Tools failed. Summarize whatever info is available.",
            })
            continue

        for res in valid_results:
            messages.append({
                "role": "tool",
                "name": res["name"],
                "content": serialize_content(res["content"]), 
            })


# Entry
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    q = input("Ask something: ")
    print(agent(q))