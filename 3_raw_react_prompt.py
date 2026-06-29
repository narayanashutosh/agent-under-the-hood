from dotenv import load_dotenv
import ollama
from langsmith import traceable
import re
import inspect

load_dotenv()

MAX_ITERATION = 10
MODEL = "qwen3:1.7b"

# ------------- Tools (Langchain @tools decorator) ------------

@traceable(run_type="tool")
def get_product_price(product: str)->float:
    """
    Lookup the price of a product in the catalog.
    Args:
        product: The product name, e.g. 'laptop', 'headphones', 'keyboard'    
    Returns:
        The price of the product or 0 if not found.
    """
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier:str)->float:
    """
    Apply a discount tier to a price and return the final price.
    Args:
        price: The original price of the product.
        discount_tier: The discount tier: 'bronze', 'silver', or 'gold'.    
    Returns:
        The discounted price of the product.
    """
    print(f"    >> Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1-discount/100), 2)

tools = {
    "get_product_price":get_product_price,
    "apply_discount":apply_discount
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
    STRICT RULES - You must follow these exactly:
    1.NEVER guess or assume any product price. You MUST call get_product_price first to get the real price. 
    2. Only call apply_discount AFTER you have received a price from get_pdouct_price. Pass the exact price return by get_product_price - do NOT pass a made-up number. 
    3. NEVER calculate discount yourself using math. Always use the apply_discount tool.
    4. If the user does not specify a discount tier. ask them which tier to use - do NOT assume one.

    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {{question}}
    Thought:
"""

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traces(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

# ------------- Agent Loop -------------

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\n--- Iteration {iteration} ---")

        full_prompt = prompt + scratchpad

        response = ollama_chat_traces(
            model=MODEL,
            messages = [{"role": "user", "content": full_prompt}],
            options = { "stop": ["\nObservation"], "temperature": 0 }
        )
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        # If no tool calls this is the final answer
        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"   [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # Direct function call
        observation = tool_to_use(**tool_args)

        print(f"   [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            { "role":"tool", "content": str(observation) }
        )
    print("ERROR: Max iterattions reached without a final answer")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    
    question = "What is the price of a laptop after applying a gold discount?"
    print()
    result = run_agent(question)