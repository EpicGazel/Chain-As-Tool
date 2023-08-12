from CustomTool import CustomTool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Callable
from dotenv import load_dotenv
load_dotenv()


def create_chain_function(prompt_template: PromptTemplate, model_name: str = "gpt-3.5-turbo",
                          temperature: float = 0.7) -> Callable[[str], str]:
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    chain = LLMChain(prompt=prompt_template, llm=llm)
    return lambda input_str: chain.run(input_str)


def create_chain_tool(name: str, description: str, prompt_template: PromptTemplate,
                            model_name: str = "gpt-3.5-turbo",
                            temperature: float = 0.7) -> CustomTool:
    """
        Create a chain tool that utilizes a language model to execute a LLMChain on an input string.

        Parameters:
            name (str): The name of the chain tool.
            description (str): A brief description of what the chain tool does.
            prompt_template (PromptTemplate): An instance of PromptTemplate that defines the instructions for the chain.
            model_name (str, optional): The name of the language model to use. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): The temperature value for language model sampling. Defaults to 0.7.

        Returns:
            CustomTool: A CustomTool instance with the specified name, description, and run_function. Inherits from BaseTool.

        Example:
            # Define a PromptTemplate for the pun converter chain tool
            # The PromptTemplate consists of a template string that contains placeholder '{prompt}'
            # The '{prompt}' placeholder will be replaced with the input text during execution
            # The chain tool will convert the input text into puns using the language model
            prompt_template = PromptTemplate.from_template(
                "Add in and convert as much of the input into puns as you can.\n" +
                "Input: {prompt}\n" +
                "Answer: This prompt in pun talk is: "
            )

            # Create the pun converter chain tool using the provided prompt template and default model settings
            pun_converter = create_chain_tool(
                name="pun-converter",
                description="Converts input text to puns",
                prompt_template=prompt_template
            )
    """
    chain_func = create_chain_function(prompt_template, model_name, temperature)
    return CustomTool(name, description, chain_func)


# Run a linear chain of prompts with models to get desired result
def create_chain_tool_multi(name: str, description: str, details: list):
    functions = []
    for params in details:
        if len(params) < 2:
            params.append("gpt-3.5-turbo")
        if len(params) < 3:
            params.append(0.7)

        functions.append(create_chain_function(params[0], params[1], params[2]))

    def all_functions(input_str: str) -> str:
        print(f"Input string: {input_str}")
        for function in functions:
            input_str = function(input_str)

        print(f"Output string: {input_str}")
        return input_str

    return CustomTool(name, description, all_functions)


# Test Tools
uppercase_tool = CustomTool("uppercase-tool", "Converts input text to uppercase",
                            lambda input_str: f"The string in uppercase is: {input_str.upper()}")

legal_talk = create_chain_tool("legal-talk", "Converts input text to legal talk",
                               PromptTemplate.from_template("Input: {prompt}\nAnswer: This prompt in legal talk is: "))

pun_talk = create_chain_tool("pun-converter", "Converts input text to puns",
                             PromptTemplate.from_template(
                                 "Add in and convert as much of the input into puns as you can.\n" +
                                 "Input: {prompt}\n" +
                                 "Answer: This prompt in pun talk is: "))

multi_test = create_chain_tool_multi("cool-function", "Make the input text much cooler.", [
    [PromptTemplate.from_template("Make the input into a rap.\nInput: {prompt}\nRap: ")],
    [PromptTemplate.from_template("Format the input into stanzas using \n characters.\nInput: {prompt}\nAnswer: ")],
    [PromptTemplate.from_template("Add some catchy zingers in.\nInput: {prompt}\nAnswer: ")],
])