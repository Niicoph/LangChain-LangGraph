from dotenv import load_dotenv

load_dotenv()


from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# ReAct agent -> looping controller that simulates reasoning
# Thought: "I should search LinkedIn."
# Action: TavilySearch("AI jobs")
# Observation: (search results)
# Thought: "Now I should summarize."
# Action: "Write final answer.


tools = [TavilySearch()]

# Since ChatOpenAI automatically injects a "stop" tokens parameter when building requests for agents, models that don't support stop sequences won't work
# OpenAI's API rejects the request -> hence 400 BadRequestError.
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# Why the stop? because it tells the agent to stop after a string is matched. Then, Agent can (thought -> action -> input) and decide the next move.

react_prompt = hub.pull("hwchase17/react")


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)


chain = agent_executor


def main():

    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in europe on linkedin and list their details",
        }
    )

    print(result)


if __name__ == "__main__":

    main()
