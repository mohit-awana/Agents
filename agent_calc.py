import ollama
from pydantic import BaseModel, ValidationError
from typing import Dict, Literal, Type


# ── Tool ─────────────────────────────────────────────────────────────────────

class CalculateTool(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float

    @classmethod
    def description(cls) -> str:
        return "Perform basic arithmetic on two numbers."

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

    def run(self) -> float:
        match self.operation:
            case "add":      return self.a + self.b
            case "subtract": return self.a - self.b
            case "multiply": return self.a * self.b
            case "divide":
                if self.b == 0:
                    raise ValueError("Division by zero")
                return self.a / self.b


# ── Registry ──────────────────────────────────────────────────────────────────

TOOLS: Dict[str, Type[BaseModel]] = {
    CalculateTool.__name__: CalculateTool,
}


# ── Agent ─────────────────────────────────────────────────────────────────────

MODEL  = "llama3.1:latest"
SYSTEM = (
    "You are a strict calculator assistant. "
    "You ONLY respond by calling the calculate tool. "
    "If the input contains an unrecognized or unsupported operation, "
    "respond with exactly: 'Unsupported operation.' "
    "NEVER guess, infer, or answer without calling the tool first."
)

SUPPORTED = {"add", "subtract", "multiply", "divide", "+", "-", "*", "/"}

def agent(user_input: str) -> str:
    # if not any(op in user_input for op in SUPPORTED):
    #     print("Agent: Unsupported operation.")
    #     return "Unsupported operation."
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_input},
    ]
    print(f"\nUser: {user_input}")

    while True:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=[t.model_json_schema() for t in TOOLS.values()],
        )
        msg = response.message
        messages.append(msg)

        if not msg.tool_calls:
            print(f"Agent: {msg.content}")
            return msg.content

        for call in msg.tool_calls:
            tool_name = call.function.name
            tool_cls = TOOLS.get(call.function.name)
            if tool_cls is None:
                print(f"Sorry, I've limited toolds, I don’t support '{tool_name}'. Please try another question.")

                #raise KeyError(f"Unknown tool: {call.function.name}")

            try:
                result = tool_cls.model_validate(call.function.arguments).run()
                print(f"  [tool] {call.function.name}({call.function.arguments}) → {result}")
                messages.append({"role": "tool","name": call.function.name,"content": str(result)})

            except (ValidationError, ValueError) as e:
                messages.append({"role": "tool", "content": f"Error: {e}"})
                messages.append({
                    "role": "system",
                    "content": "The tool call failed. Inform the user their request is unsupported.",
                })


if __name__ == "__main__":
    question = input("Please ask an arithmetic question: ")
    print(agent(question))