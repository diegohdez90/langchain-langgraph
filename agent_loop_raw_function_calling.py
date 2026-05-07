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


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'"
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price"
                    },
                    "discount_type": {
                        "type": "string",
                        "description": "discount tier: 'bronze', 'silver' and 'gold'"
                    }
                },
                "required": ["price", "discount_type"]
            }
        }
    }
]


@traceable(name='Ollama Chat', run_type='llm')
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Ollsama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        'get_product_price': get_product_price,
        'apply_discount': apply_discount
    }

    print(f"Question: {question}")
    print("=" * 60)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRIC RULES - you must follow exactly:\n"
                "1. NEVER guess or assume any product price. ",
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discount yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use(**tool_args)
        print(f"[Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation)
            }
        )
    
    print(
        "ERROR: MAX_ITERATIONS reached without a final answer"
    )
    
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")

    result = run_agent(
        question="What is the price of a laptop after applying a gold discount"
    )
