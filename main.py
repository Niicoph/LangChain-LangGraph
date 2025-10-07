from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain import hub
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch  # Tavily Search tool

load_dotenv()

# A create_react_agent is a function that creates a react agent. This takes a few parameters such as: LLM, tools, prompt, output_parser, etc.
# The AgentExecutor is the runtime. It is responsible for executing the agent. Actually runs the tools when the agent decides to use them.

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")

# Receives tools , tools_names and input question by the user
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chain = agent_executor


def main():
    result = chain.invoke(
        {"input": "Search for 3 job postings for an AI engineer in Europe, particularly in Spain. List their details."})

    print(result)


if __name__ == "__main__":
    main()
