from dotenv import load_dotenv
import ollama
from langsmith import traceable

load_dotenv()

MAX_ITERATION = 10
MODEL = "qwen3:1.7b"

# ------------- Tools (Langchain @tools decorator) -------------

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

tools_for_llm = [
    {
      "type": "function",
      "function": {
        "name": "get_product_price",
        "description": "Lookup the price of a product in the catalog.",
        "parameters": {
          "type": "object",
          "properties": {
            "product": {
                "type": "string", 
                "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'."
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
                "type": "float", 
                "description": "The price of the product or 0 if not found"
            },
            "discount_tier": {
                "type": "string", 
                "description": "The discount tier: 'bronze', 'silver', or 'gold'."
            }
          },
          "required": ["price", "discount_tier"]
        }
      }
    }
]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traces(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# ------------- Agent Loop -------------

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):    
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: {question}")
    print("=" * 60)
    
    messages = [
        {
            "role": "system", 
            "content":
                    """
                    You are a helpful shopping assistant. You have access to a product catalogue tooland a discount tool.
                    STRICT RULES - You must follow these exactly:
                    1.NEVER guess or assume any product price. You MUST call get_product_price first to get the real price. 
                    2. Only call apply_discount AFTER you have received a price from get_pdouct_price. Pass the exact price return by get_product_price - do NOT pass a made-up number. 
                    3. NEVER calculate discount yourself using math. Always use the apply_discount tool.
                    4. If the user does not specify a discount tier. ask them which tier to use - do NOT assume one.
                    """
        },
        { "role":"user", "content": question }
    ]
    
    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\n--- Iteration {iteration} ---")

        response = ollama_chat_traces(messages=messages)
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