from dotenv import load_dotenv

#For the AI to understand your second question, it needs access to your previous conversation
# MessagesState = A ready-made structure for maintaining message history in LangGraph
from langgraph.graph import MessagesState
#----------------------------------------------------------------------------

#The LLM can request a tool but cant execute it
#ToolNode executes the tool
from langgraph.prebuilt import ToolNode
#----------------------------------------------------------------------------

from react import llm, tools

load_dotenv()

# """multiline string, 
# system message gives the model high-level instructions about how it should behave
SYSTEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""
#----------------------------------------------------------------------------

#defining your first LangGraph node
# state: MessagesState expects a LangGraph state containing messages
def run_agent_reasoning(state: MessagesState) -> MessagesState:

    #describes what the function is responsible for
    """
    Run the agent reasoning node.
    """
    #----------------------------------------------------------------------------

    #.invoke() sends an input to the model and waits for a response
    # * takes all elements inside an iterable and unpacks them
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    #----------------------------------------------------------------------------
    return {"messages": [response]}

#ToolNode reads that tool call
tool_node = ToolNode(tools)
#----------------------------------------------------------------------------