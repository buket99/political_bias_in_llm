import json

def main():
    from wahlomat_agent import WahlOMatAgent

    agent = WahlOMatAgent(model="gpt-4", temperature=0.2)

    # Path to English questions
    quesitons_file_en = "/Users/buketkurtulus/Desktop/HM/Semester 5/Hauptseminar/Experiment/questions/questions_eng.json"
    # Path to German questions
    quesitons_file_en = "/Users/buketkurtulus/Desktop/HM/Semester 5/Hauptseminar/Experiment/questions/questions_de.json"

    # Run once per question
    num_runs = 10

    print(f"Running LLM queries for {num_runs} iterations per question...")
    agent.run_on_questions(quesitons_file_en, num_runs=num_runs)

    # save results
    with open("llm_responses.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done. Results saved to llm_responses.json")

if __name__ == "__main__":
    main()