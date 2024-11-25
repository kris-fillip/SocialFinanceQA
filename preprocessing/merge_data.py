import json
import os
import random

from config.constants import CONSIDER_GILDING_AND_AWARDING, PERCENTILES, SUBREDDITS, MAX_LEVEL
from pushshift import read_lines_zst


def add_data(file_path: str, data: list[dict], subreddit: str) -> list[dict]:
    file_size = os.stat(file_path).st_size
    file_lines = 0
    bad_lines = 0

    for line, file_bytes_processed in read_lines_zst(file_path):
        file_lines += 1
        if file_lines % 100000 == 0:
            print(
                f"Line: {file_lines:,} Bad Lines: {bad_lines:,} Bytes Processed: {file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
        try:
            obj = json.loads(line)
            data.append(obj)

        except (KeyError, json.JSONDecodeError) as err:
            print("Error:" + err)
    print(f"Subreddit {subreddit} has {file_lines} entries")
    return data


def merge_data() -> None:
    for curated_or_not in CONSIDER_GILDING_AND_AWARDING:
        for percentile in PERCENTILES:
            current_directory = f"{curated_or_not}_percentile{percentile}"
            path = f"./data/question_answers/{current_directory}"
            result_data = []
            output_path = os.path.join(path, f"complete_qa.json")
            for subreddit in SUBREDDITS:
                current_path = os.path.join(path, f"{subreddit}_qa_max_level_{MAX_LEVEL}.zst")
                result_data = add_data(current_path, result_data, subreddit)
                random.Random(42).shuffle(result_data)
            print(f"QA Dataset for {current_directory} has {len(result_data)} entries.")
            with open(output_path, "w") as f:
                json.dump(result_data, f)

if __name__ == "__main__":
    merge_data()
