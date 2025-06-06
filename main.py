import json
from wahlomat_agent.agent import WahlOMatAgent
from modules.LLMWrapper import ChatGPTWrapper
from dotenv import load_dotenv
import json
load_dotenv()


def main():

    models = {
        "gpt-3.5": ChatGPTWrapper(model="gpt-3.5-turbo")
        #"grok": GrokWrapper(),
        #"deepseek": DeepSeekWrapper()
    }

    #########################################################
    #################   TESTING PURPOSES    #################
    #########################################################
    # Only use English questions
    # questions_file = "/Users/buketkurtulus/Desktop/HM/Semester 5/Hauptseminar/Programmierung/questions/questions_eng.json"
    # lang = "en"  # used for output file naming
    # num_runs = 3
    #
    # # Choose model(s) — you can later extend this again
    # models = {
    #     "gpt-3.5": ChatGPTWrapper(model="gpt-3.5-turbo")
    # }
    #
    # with open(questions_file, "r", encoding="utf-8") as f:
    #     questions = json.load(f)["questions"]
    #
    # for model_name, wrapper in models.items():
    #     print(f"Running {model_name} on {lang.upper()} questions...")
    #
    #     agent = WahlOMatAgent(wrapper)
    #     results = agent.run_on_questions(questions_file, num_runs=num_runs)
    #
    #     # Save to TXT
    #     print("Done. Start saving responses...")
    #     output_txt = f"{model_name}_{lang}_responses.txt"
    #     with open(output_txt, "w", encoding="utf-8") as f:
    #         for qid, entry in results.items():
    #             f.write(f"Q{qid}: {entry['question']}\n")
    #             for i, raw in enumerate(entry["raw_responses"]):
    #                 f.write(f"  Run {i + 1}: {raw}\n")
    #             f.write("\n")
    #
    #     print(f"Saved: {output_txt}")

    question_files = {
         "english": "/Users/buketkurtulus/Desktop/HM/Semester 5/Hauptseminar/Programmierung/questions/questions_eng.json",
         "german": "/Users/buketkurtulus/Desktop/HM/Semester 5/Hauptseminar/Programmierung/questions/questions_de.json"
    }

    num_runs = 1 # number of runs to test

    for lang_label, file_path in question_files.items():
        if "questions_de" in file_path.lower():
            lang_code = "de"
        elif "questions_eng":
            lang_code = "en"
        else:
            raise ValueError(f"Unknown language label")

        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)["questions"]

        for model_name, wrapper in models.items():
            print(f"lang_label: {lang_label}, lang_code: {lang_code}")
            print(f"Running {model_name} on {lang_code.upper()} questions...")
            agent = WahlOMatAgent(wrapper)
            results = agent.run_on_questions(file_path, num_runs=num_runs, lang=lang_code)

            print("Done. Start saving responses...")
            output_txt = f"{model_name}_{lang_label}_responses.txt"
            with open(output_txt, "w", encoding="utf-8") as f:
                for qid, entry in results.items():
                    f.write(f"Q{qid}: {entry['question']}\n")
                    for i, raw in enumerate(entry["raw_responses"]):
                        f.write(f"  Run {i + 1}: {raw}\n")
                        # f.write("\n")
            print(f"Saved: {output_txt}")


if __name__ == "__main__":
    main()