import csv

RESPONSE_MAP = {
    "disagree": 0,
    "neutral": 1,
    "agree": 2
}




file_path = "/modules/gpt-3.5_english_responses.txt"

convert_txt_to_csv(file_path, "gpt-3.5_english_responses_converted.csv")