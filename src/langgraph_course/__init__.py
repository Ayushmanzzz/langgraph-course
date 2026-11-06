# react -> nodes -> init

from dotenv import load_dotenv

#to represent the initial user input when starting your graph
from langchain_core.messages import HumanMessage
#----------------------------------------------------------------------------

#StateGraph is a graph builder, END is special LangGraph constant representing the end of graph execution
from langgraph.graph import MessagesState, StateGraph, END
#----------------------------------------------------------------------------

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason"
ACT= "act"
LAST = -1

#function that decides where the graph should go next
def should_continue(state: MessagesState) -> str:
    #["messages"][LAST] Gets the most recent message
    if not state["messages"][LAST].tool_calls:
        return END
    #Move to the action node and execute the requested tool
    return ACT
    #----------------------------------------------------------------------------
#----------------------------------------------------------------------------

#StateGraph(MessagesState) says This graph will use MessagesState to manage its data
#graph builder and not a complete executable graph
flow = StateGraph(MessagesState)
#----------------------------------------------------------------------------

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

#After the agent_reason node finishes, evaluate the condition
#decide whether to act or end
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})
#----------------------------------------------------------------------------

flow.add_edge(ACT, AGENT_REASON)

#flow = Graph builder.
#app = Executable graph.

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)


# User input(app.invoke()) -> Agent reasoning(run_agent_reasoning(state)) -> Routing decision(should_continue(state)) -> Tool Execution(tool_node) -> Agent reasoning again -> Tool execution agaain -> Final reasoning -> Final response