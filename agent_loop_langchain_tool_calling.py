from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langsmith import traceable

from .constants import (
    MAX_ITERATIONS,
    MODEL
)

load_dotenv()


@tool
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


@tool
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

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [
        get_product_price,
        apply_discount
    ]

    tools_dict = {
        t.name for t in tools
    }

    llm = init_chat_model(
        f"ollama: {MODEL}", temperature=0
    )
    llm_with_tools = llm.bind_tools(tools=tools)
    print(f"Question: {question}")
    print("=" * 60)
    messages = [
        SystemMessage(
            content=(
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
        ),
        HumanMessage(content=question),
    ]
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id')
        
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use.invoke(tool_args)
        print(f"[Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id
            )
        )
    
    print(
        "ERROR: MAX_ITERATIONS reached without a final answer"
    )


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")

    result = run_agent(
        question="What is the price of a laptop after applying a gold discount"
    )
