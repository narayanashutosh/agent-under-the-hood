from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable
import os

load_dotenv()

MAX_ITERATION = 10
MODEL = "qwen3:1.7b"
AZURE_LLM_MODEL = ""



# ------------- Tools (Langchain @tools decorator) -------------

@tool
def get_product_price(product: str)->float:
    """
    Lookup the price of a product in the catalog.
    """
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier:str)->float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    print(f"    >> Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1-discount/100), 2)

# ------------- Agent Loop -------------

@traceable(name="Langchain Agent Loop")
def run_agent(question: str, llm_type: str = "AZURE_OPENAI"):
    model_name = f"ollama:{MODEL}"
    llm_kwargs = {}

    if llm_type=="AZURE_OPENAI":
        model_name = f"azure_openai:{os.environ.get('OPENAI_MODEL')}"
        llm_kwargs = {
                            "azure_endpoint": os.environ.get("OPENAI_API_ENDPOINT"),
                            "api_version": os.environ.get("OPENAI_API_VERSION"),
                            "azure_deployment": os.environ.get("OPENAI_DEPLOYMENT"),
                            "api_key": os.environ.get("OPENAI_API_KEY"),
                        }

    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(model_name, temperature=0, **llm_kwargs)
    llm_with_tools = llm.bind_tools(tools)
    print(f"LLM provider: {llm_type}")
    print("=" * 60)
    print(f"Question: {question}")
    print()
    
    messages = [
        SystemMessage(
            content=(
                """
                You are a helpful shopping assistant. You have access to a product catalogue tooland a discount tool.
                STRICT RULES - You must follow these exactly:
                1.NEVER guess or assume any product price. You MUST call get_product_price first to get the real price. 
                2. Only call apply_discount AFTER you have received a price from get_pdouct_price. Pass the exact price return by get_product_price - do NOT pass a made-up number. 
                3. NEVER calculate discount yourself using math. Always use the apply_discount tool.
                4. If the user does not specify a discount tier. ask them which tier to use - do NOT assume one.
                """
            )
        ),
        HumanMessage(content=question)        
    ]

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # If no tool calls this is the final answer
        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"   [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"   [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print("ERROR: Max iterattions reached without a final answer")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    
    question = "What is the price of a laptop after applying a gold discount?"
    print()
    result = run_agent(question, "AZURE_OPENAI")
    print()
    result = run_agent(question, "OLLAMA")