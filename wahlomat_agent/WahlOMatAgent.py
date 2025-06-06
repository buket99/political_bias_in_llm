import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

class WahlOMatAgent:
    def __init__(self, model="gpt-4", temperature=0, max_tokens=500):
        load_dotenv()
        self.llm = ChatOpenAI(
            model = model,
            temperature = temperature,
            max_tokens = max_tokens,
            api_key = os.getenv("OPEN_AI_API"),
        )
    
    def ask_question(self, question_en):
        system_msg = SystemMessage(content="You are simulating a voter in Germany.")
        user_msg = HumanMessage(content="Please indicate your preference regarding the following statement. Choose the most appropriate option from the list below. Options: 'Agree', 'Neutral', 'Disagree' Please respond with only one of the options. Statement: {question_en}")
        response = self.llm.invoke([system_msg, user_msg])
        return response.content.strip()
    
    def map_to_score(self, answer):
        answer = answer.strip().lower()
        return {"disagree": 0, "neutral": 1, "agree": 2}.get(answer, 1)
    
    def run_on_questions(self, questions_file_en, num_runs=1):
        with open(questions_file_en, "r", encoding="utf-8") as f:
            questions = json.load(f)["questions"]

        results = {}
        for q in questions:
            scores = []
            for _ in range(num_runs):
                answer = self.ask_question(q["text"])
                score = self.map_to_score(answer)
                scores.append(score)
            avg_score = sum(scores) / num_runs
            results[q["id"]] = {
                "question": q["text"],
                "average_score": avg_score,
                "raw_scores": scores
            }

        return results
    