from langchain import PromptTemplate
from langchain.agents import ZeroShotAgent, OpenAIFunctionsAgent, AgentExecutor, load_tools
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
import traceback
import ChainTools
from dotenv import load_dotenv
load_dotenv()


def main():
    langchain.verbose = True

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2) #max_tokens=250
    tools = load_tools(["llm-math", "wikipedia", "ddg-search"], llm)
    # tools += [ChainTools.uppercase_tool, ChainTools.legal_talk, ChainTools.pun_talk]
    tools += [ChainTools.multi_test]

    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    system_prompt = SystemMessagePromptTemplate(prompt=PromptTemplate.from_template(
        "Your name is {name}. The current date is November 2nd, 2016.\n"))

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, HumanMessagePromptTemplate(prompt=prompt)])

    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", input_key="input")

    functions_agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=chat_prompt)
    functions_agent_chain = AgentExecutor.from_agent_and_tools(agent=functions_agent, tools=tools,
                                                               memory=memory, handle_parsing_errors=True)

    try:
        while True:
            response = functions_agent_chain.run(name="James", input=input("Enter input: "))
            print(response)
    except Exception:
        traceback.print_exc()
        print("Exiting...")


if __name__ == "__main__":
    main()
