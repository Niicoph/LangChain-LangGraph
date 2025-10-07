from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch  # Tavily Search tool

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

# A create_react_agent is a function that creates a react agent. This takes a few parameters such as: LLM, tools, prompt, output_parser, etc.
# The AgentExecutor is the runtime. It is responsible for executing the agent. Actually runs the tools when the agent decides to use them.

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")

# Returns the output formatted to the AgentResponse schema
structured_llm = llm.with_structured_output(AgentResponse)

react_prompt = hub.pull("hwchase17/react")

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# extract the output from the agent
extract_output = RunnableLambda(lambda x: x("output"))

chain = agent_executor | extract_output | structured_llm


def main():
    result = chain.invoke(
        {
            "input": "Search for 3 job postings for an AI engineer in Europe, particularly in Spain. List their details."
        }
    )

    print(result)


if __name__ == "__main__":
    main()
