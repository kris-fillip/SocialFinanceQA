import argparse
import json
import os
import random
import zstandard
from datetime import datetime
from tqdm import tqdm

from config.constants import SUBREDDITS, CONSIDER_GILDING_AND_AWARDING, PERCENTILES, MAX_LEVEL
from pushshift import read_lines_zst, write_line_zst
from utils.entry_creation import collect_merged_information_two_answers


def create_question_answer_dataset(file_path: str, subreddit: str, answer2_selectors_set: set[str], current_directory: str, reproduction: bool, max_level: int) -> int:
    amount_of_lines = len(list(read_lines_zst(file_path)))
    file_size = os.stat(file_path).st_size
    file_lines = 0
    created = None
    bad_lines = 0
    lines_created = 0
    not_enough_comments = 0
    score_difference_small = 0
    level_string = f"_max_level_{max_level}"
    output_path = f"./data/question_answers/{current_directory}/"
    os.makedirs(output_path, exist_ok=True)
    print("created directory: " + output_path)
    output_path = os.path.join(output_path, f"{subreddit}_qa{level_string}.zst")
    handle = zstandard.ZstdCompressor().stream_writer(open(output_path, 'wb'))


    for line, file_bytes_processed in tqdm(read_lines_zst(file_path), total=amount_of_lines):
        file_lines += 1
        if file_lines % 100000 == 0:
            print(
                f"{created} Line: {file_lines:,} Bad Lines: {bad_lines:,} Bytes Processed: {file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(int(obj["created_utc"])).strftime("%Y/%m/%d")
            submission_id = obj["id"]
            obj_map = {submission_id: {"level": 0, "object": obj, "parent": None}}
            if "comments" not in obj:
                continue
            handled_idxs = []
            obj_map_size = len(obj_map)
            # Construct an object map mapping object ids to objects, their level and parent id
            while True:
                if len(handled_idxs) >= len(obj["comments"]):
                    break
                for idx, comment in enumerate(obj["comments"]):
                    if idx in handled_idxs:
                        continue
                    comment_parent = comment["parent_id"]
                    if comment_parent in obj_map:
                        parent_level = obj_map[comment_parent]["level"]
                        if parent_level > max_level:
                            continue
                        comment_id = comment["name"]
                        if comment_id is None:
                            comment_id = "t1_" + comment["id"]
                        obj_map[comment_id] = {
                            "level": parent_level + 1,
                            "object": comment,
                            "parent": comment_parent
                        }
                        handled_idxs.append(idx)
                if len(obj_map) == obj_map_size and obj_map_size != len(obj["comments"]) + 1:
                    # print(f"{len(obj['comments']) - len(obj_map) + 1} unconnected comment(s) found")
                    break
                obj_map_size = len(obj_map)
            # Iterate over object map in order to generate question answer pairs and write them to file
            cur_obj_list = list(obj_map.values())
            cur_obj_list = [obj for obj in cur_obj_list if obj["parent"] == submission_id]
            parent = obj_map[submission_id]
            cur_obj_list = [obj for obj in cur_obj_list if obj["object"]["author"] !=
                            parent["object"]["author"]]  # Filter out authors answering themselves directly
            if len(cur_obj_list) < 2:
                not_enough_comments += 1
                continue
            cur_obj_list.sort(key=lambda el: el["object"]["score"], reverse=True)
            cur_scores_list = [el["object"]["score"] for el in cur_obj_list]
            best_score = cur_scores_list[0]
            qualified_bad_answer_indices = [idx for idx, value in enumerate(
                cur_scores_list) if (best_score - value >= 10) and (value <= 3)]
            if len(qualified_bad_answer_indices) == 0:
                score_difference_small += 1
                continue
            if reproduction:
                # The ids from the answer2_selectors file are used to reproduce the initially randomly selected non-preferred answers in the dataset creation.
                # This is necessary in order to reliably reproduce the same dataset since the seed for the random.choice was missing in the initial dataset creation.
                found = False
                for obj_list_idx in qualified_bad_answer_indices:
                    cur_obj_to_handle = cur_obj_list[obj_list_idx]["object"]
                    cur_obj_id_to_handle = cur_obj_to_handle["id"]
                    if cur_obj_id_to_handle in answer2_selectors_set:
                        selected_index = obj_list_idx
                        found = True
                        break
                if found == False:
                    print("Reproduction Error! There should always be a match")
                    selected_index = random.Random(42).choice(qualified_bad_answer_indices)
            else:
                # In case an explicit reproduction of the dataset is not desired / a new dataset should be created, the non-preferred answers are now selected through a seeded random.choice().
                selected_index = random.Random(42).choice(qualified_bad_answer_indices)
            new_obj = collect_merged_information_two_answers(parent["object"], cur_obj_list[0]["object"], cur_obj_list[selected_index]["object"], subreddit)
            new_line = json.dumps(new_obj)
            write_line_zst(handle, new_line)
            lines_created += 1

        except (KeyError, json.JSONDecodeError) as err:
            print("Error:" + err)

    handle.close()
    print(f"Lines created: {lines_created} out of {file_lines} submissions of subreddit {subreddit}")
    print(f"{not_enough_comments} entries with less than two comments available")
    print(f"{score_difference_small} entries available score difference smaller 10")
    return lines_created

def preference_matching(reproduction: bool) -> None:
    total_lines = 0
    with open("./data/reproduction/answer2_selectors.json", "r") as inputfile:
        answer2_selectors_set = set(json.load(inputfile))
    for curated_or_not in CONSIDER_GILDING_AND_AWARDING:
        for percentile in PERCENTILES:
            current_directory = f"{curated_or_not}_percentile{percentile}"
            for subreddit in SUBREDDITS:
                current_path = f"./data/filtered/{current_directory}/{subreddit}_filtered_combined.zst"
                total_lines += create_question_answer_dataset(current_path, subreddit, answer2_selectors_set, current_directory, reproduction, max_level=MAX_LEVEL)
    print(f"total dataset size: {total_lines}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("preference_matching")
    parser.add_argument("-rep", "--reproduction", help="Boolean indicating if original dataset should be reproducted",
                        default=True, action="store_false")
    args = parser.parse_args()
    preference_matching(reproduction=args.reproduction)