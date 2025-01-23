# Install required packages
!pip install -U langchain_community langchain-openai langchain-anthropic langchain langgraph bs4

import getpass
import os
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langsmith.evaluation import evaluate

# Set environment variables for API keys
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("ANTHROPIC_API_KEY")

# Load LCEL documentation
url = "https://python.langchain.com/docs/concepts/lcel/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# Sort and concatenate content
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in d_reversed])

# Set up LangChain prompts for OpenAI and Anthropic
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a coding assistant with expertise in LCEL. Here is the LCEL documentation: {context}. Answer the user question based on the above provided documentation. Ensure any code you provide can be executed with all required imports and variables defined."""), 
        ("placeholder", "{messages}")
    ]
)

# Data model for code generation output
class Code(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

# Set up LLM for OpenAI
llm = ChatOpenAI(temperature=0, model="gpt-4")
code_gen_chain_oai = code_gen_prompt | llm.with_structured_output(Code)

# Define retry and fallback logic for code generation
def check_claude_output(tool_output):
    if tool_output["parsing_error"]:
        raw_output = str(tool_output["raw"].content)
        error = tool_output["parsing_error"]
        raise ValueError(f"Error parsing output! {raw_output}. Parse error: {error}")
    elif not tool_output["parsed"]:
        raise ValueError("Failed to invoke tool! Be sure to invoke the tool to structure the output.")
    return tool_output

# Define LangGraph state and graph
class GraphState(TypedDict):
    error: str
    messages: list
    generation: str
    iterations: int

# Define workflow for generating and validating code solutions
def generate(state: GraphState):
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    if error == "yes":
        messages.append(("user", "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:"))
    
    code_solution = code_gen_chain_oai.invoke({"context": concatenated_content, "messages": messages})
    messages.append(("assistant", f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}"))
    iterations += 1

    return {"generation": code_solution, "messages": messages, "iterations": iterations}

def code_check(state: GraphState):
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    imports = code_solution.imports
    code = code_solution.code

    try:
        exec(imports)
    except Exception as e:
        messages.append(("user", f"Your solution failed the import test: {e}"))
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}

    try:
        exec(imports + "\n" + code)
    except Exception as e:
        messages.append(("user", f"Your solution failed the code execution test: {e}"))
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "yes"}

    return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": "no"}

def reflect(state: GraphState):
    messages = state["messages"]
    code_solution = state["generation"]

    reflections = code_gen_chain_oai.invoke({"context": concatenated_content, "messages": messages})
    messages.append(("assistant", f"Here are reflections on the error: {reflections}"))
    
    return {"generation": code_solution, "messages": messages, "iterations": state["iterations"]}

def decide_to_finish(state: GraphState):
    if state["error"] == "no" or state["iterations"] >= 3:
        return "end"
    else:
        return "generate"

workflow = StateGraph(GraphState)
workflow.add_node("generate", generate)
workflow.add_node("check_code", code_check)
workflow.add_node("reflect", reflect)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges(
    "check_code", decide_to_finish, {"end": END, "reflect": "reflect", "generate": "generate"}
)
workflow.add_edge("reflect", "generate")

app = workflow.compile()

# Test question
question = "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
solution = app.invoke({"messages": [("user", question)], "iterations": 0, "error": ""})

# Evaluate performance
client = langsmith.Client()
public_dataset = "https://smith.langchain.com/public/326674a6-62bd-462d-88ae-eea49d503f9d/d"
client.clone_public_dataset(public_dataset)

code_evalulator = [check_import, check_execution]
dataset_name = "lcel-teacher-eval"

experiment_results_ = evaluate(
    predict_base_case,
    data=dataset_name,
    evaluators=code_evalulator,
    experiment_prefix=f"test-without-langgraph-gpt-4",
    max_concurrency=2,
    metadata={"llm": "gpt-4"},
)

experiment_results = evaluate(
    predict_langgraph,
    data=dataset_name,
    evaluators=code_evalulator,
    experiment_prefix=f"test-with-langgraph-gpt-4",
    max_concurrency=2,
    metadata={"llm": "gpt-4"},
)

