# Chain-As-Tool

Chain-As-Tool is a Python library that provides a set of tools for creating and executing linear chains of prompts with language models. It simplifies the process of interacting with language models and allows for the creation of complex conversational agents.

### Usage Example

To use Chain-As-Tool, follow these steps (some code is excluded, check the using_chaintool.py file for all of it):

1. Install the required dependencies by running `pip install langchain dotenv openai`.

2. Import the necessary modules and classes:

```python
from langchain import PromptTemplate
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
```

3. Load the environment variables from a `.env` file using `load_dotenv()`.

4. Define the necessary prompt templates and chain-tool for your conversational agent. For example:

```python
prompt_template = PromptTemplate.from_template("Add in and convert as much of the input into puns as you can.\n" +
                                               "Input: {prompt}\n" +
                                               "Answer: This prompt in pun talk is: ")

pun_converter = create_chain_tool("pun-converter", "Converts input text to puns", prompt_template)
```

5. Create the llm, tools, and memory objects:

```python
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
tools = load_tools(["llm-math", "wikipedia", "ddg-search"], llm)
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", input_key="input")
```

6. Create the agent executor:

```python
functions_agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=chat_prompt)
functions_agent_chain = AgentExecutor.from_agent_and_tools(agent=functions_agent, tools=tools,
                                                           memory=memory, handle_parsing_errors=True)
```

7. Start the conversation loop:

```python
try:
    while True:
        response = functions_agent_chain.run(name="James", input=input("Enter input: "))
        print(response)
except Exception:
    traceback.print_exc()
    print("Exiting...")
```

This is a basic example of how to use Chain-As-Tool to create a conversational agent with GPT-3.5 autocompletion. You can customize and extend the functionality according to your specific requirements.

Feel free to create issues and contribute to the Chain-As-Tool project for additional features or bug fixes.

Please note that this is a simplified example and may not cover all the details and functionalities of Chain-As-Tool. It's recommended to refer to the official langchain documentation and examples for a comprehensive understanding of using tools.
