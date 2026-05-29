import re
import inspect

from dotenv import load_dotenv
import ollama
from langsmith import traceable

from .constants import (
    MAX_ITERATIONS,
    MODEL
)

load_dotenv()


@traceable(run_type='tool')
def get_product_price(
    product: str
) -> float:
    print(f"Launching get product price for {product}")

    prices = {
        "laptop": 1299.99,
        "headphones": 149.99,
        "keyboard": 89.99
    }

    return prices.get(product, 0)


@traceable(run_type='tool')
def apply_discount(
    price: float,
    discount_tier: str
) -> float:
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = {
    'get_product_price': get_product_price,
    'apply_discount': apply_discount
}


def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""

@traceable(name='Ollama Chat', run_type='llm')
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


@traceable(name="Ollsama Agent Loop")
def run_agent(question: str):


    print(f"Question: {question}")
    print("=" * 60)
    
    prompt = react_prompt.format(question=question)
    scratchpad = ""


    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": full_prompt
            }],
            options={
                "stop": [
                    "\nObservation"
                ],
                "temperature": 0
            }
        )
        output = response.message.content
        print(f"LLM Output: \n{output}")
        
        print("  [Parsing] Looking for Final Asnwer in LLM Output...")
        
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
        print("   [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        
        if not action_match or not action_input_match:
            print(
                "   [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break
        
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        
        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")
        
        raw_args = [x.strip() for x in tool_input_raw,split(',')]
        args = [x.split('=', 1)[-1].strip().strip("'\'") for x in raw_args]
        
        print(f"  [Tool Executing] {tool_name}({args})...")
        
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))
        print(f"[Tool Result] {observation}")
        scratchpad += f"{output}\nObservation: {observation}\nThought:"
    
    print(
        "ERROR: MAX_ITERATIONS reached without a final answer"
    )
    
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")

    result = run_agent(
        question="What is the price of a laptop after applying a gold discount"
    )
