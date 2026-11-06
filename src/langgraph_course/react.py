from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

#Tavily is a search service designed for applications that use LLMs
from langchain_tavily import TavilySearch
#----------------------------------------------------------------------------

# way to manage environment variables, especially configuration and secrets
load_dotenv()
#----------------------------------------------------------------------------

#A decorator that converts a Python function into a LangChain tool that an LLM can be instructed to use.
@tool
#----------------------------------------------------------------------------
def triple(num:float) -> float:

    #Docstring -> describes what the function does
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    #----------------------------------------------------------------------------

    return float(num) * 3

#max_results=1 is a configuration parameter telling Tavily to return at most one search result per search
#custom tool created earlier
tools = [TavilySearch(max_results=1), triple]
#----------------------------------------------------------------------------

#Create the ChatOllama instance
# creating a chat model interface configured to use Llama 3.2
#Temperature controls how much randomness is introduced into the model's generation
#bind_tools(tools)makes the tools available to the model. Something still needs to execute the requested tool and pass its result back to the model.
#stored in llm to send message to the llm
llm = ChatOllama(model="llama3.2:latest", temperature=0).bind_tools(tools)
#----------------------------------------------------------------------------

# Define → Collect → Bind → Invoke → Execute tool → Return result