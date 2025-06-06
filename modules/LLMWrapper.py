import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class LLMWrapper:
    def ask_questions(self, questions_eng: str) -> str:
        raise NotImplementedError


class ChatGPTWrapper(LLMWrapper):
    def __init__(self, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.getenv("OPEN_AI_API")

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
        )

    def ask_questions(self, question_text: str, lang: str = "en") -> str:
        if lang == "de":
            system_msg = SystemMessage(content="Du simulierst eine:n deutsche:n Wähler:in.")
            user_msg = HumanMessage(
                content=f"""Bitte gib deine Haltung zu folgender Aussage an.
                        Wähle nur eine der folgenden Optionen:
                        'Stimme zu', 'Neutral', 'Stimme nicht zu'.
                        Antworte bitte nur mit einer der Optionen.
                        Aussage: {question_text}"""
            )
        else:
            system_msg = SystemMessage(content="You are simulating a voter in Germany.")
            user_msg = HumanMessage(
                content=f"""Please indicate your preference regarding the following statement.
                        Choose one of the following options:
                        'Agree', 'Neutral', 'Disagree'.
                        Please respond with only one of the options.
                        Statement: {question_text}"""
            )

        return self.llm.invoke([system_msg, user_msg]).content.strip()

class GrokWrapper(LLMWrapper):
    def __init__(self, api_key=None):
        self.api_key = api_key
        print("to be added")

    def ask_questions(self, questions_eng: str) -> str:
        print("to be connected to Grok API")
        return "Neutral"


class DeepSeekWrapper(LLMWrapper):
    def __init__(self, api_key=None):
        self.api_key = api_key
        print("to be added")

    def ask_questions(self, questions_eng: str) -> str:
        print("to be connected to DeepSeek API")
        return "Neutral"
