import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from modules.LLMWrapper import LLMWrapper


class WahlOMatAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def map_to_score(self, answer):
        answer = answer.strip().lower()
        return {"disagree": 0, "neutral": 1, "agree": 2, "stimme nicht zu": 0, "neutral": 1, "stimme zu": 2}.get(answer, 1)
    
    def run_on_questions(self, questions_file_en, num_runs=1, lang="en"):
        with open(questions_file_en, "r", encoding="utf-8") as f:
            questions = json.load(f)["questions"]

        results = {}
        for q in questions:
            scores = []
            raw_responses = []
            print(f"Asking: {q['text']}")
            for _ in range(num_runs):
                raw_answer = self.llm.ask_questions(q["text"], lang=lang)
                score = self.map_to_score(raw_answer)
                scores.append(score)
                raw_responses.append(raw_answer)

            results[q["id"]] = {
                "question": q["text"],
                "raw_responses": raw_responses,
                "raw_scores": scores,
                "average_score": sum(scores) / num_runs,
            }

        return results
    