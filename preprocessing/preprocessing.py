import json
import os
from datetime import datetime

import pandas as pd
import zstandard
from pushshift import read_lines_zst, write_line_zst

from analysis import get_basic_attribute
from config.constants import SUBREDDITS

def get_amount_of_lines(file_path):
    return len(list(read_lines_zst(file_path)))


def preprocessing(current_file_name, current_path):
    file_size = os.stat(current_path).st_size
    file_lines = 0
    created = None
    bad_lines = 0
    output_path = os.path.join("./data/subreddits", current_file_name)
    handle = zstandard.ZstdCompressor().stream_writer(open(output_path, 'wb'))
    skip_count = 0
    seen_ids = []
    additional_skip_count = 0
    for line, file_bytes_processed in read_lines_zst(current_path):
        file_lines += 1
        if file_lines % 100000 == 0:
            print(
                f"{created} Line: {file_lines:,} Bad Lines: {bad_lines:,} Bytes Processed: {file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(int(obj["created_utc"])).strftime("%Y/%m/%d")
            parent_id = get_basic_attribute(obj, "parent_id")
            link_id = get_basic_attribute(obj, "link_id")
            # cur_id = get_basic_attribute(obj, "id")
            if parent_id in id_data_set:
                seen_ids.append(parent_id)
                skip_count += 1
                continue
            elif link_id in id_data_set:
                seen_ids.append(link_id)
                skip_count += 1
                continue
            # if cur_id in special_filter_ids_set:
            #     additional_skip_count += 1
            #     continue
            # Should already be handled by red flags -> very good
            # if cur_id == 'cfnch3y':
            #     obj["body"] += "We will see what happens by then"
            #     new_line = json.dumps(obj)
            #     write_line_zst(handle, new_line)
            # elif cur_id == "ei1pduk":
            #     obj["body"] = obj["body"][:-1]
            #     obj["body"] += "to a similar place."
            #     new_line = json.dumps(obj)
            #     write_line_zst(handle, new_line)
            # else:
            write_line_zst(handle, line)

        except (KeyError, json.JSONDecodeError) as err:
            print("Error:" + err)
            bad_lines += 1
    print(f"Skipped {skip_count} comments.")
    print(f"Skipped {additional_skip_count} comments for percentiles.")
    print(f"This should filter out {len(set(seen_ids))} submissions for subreddit {subreddit}")
    handle.close()
    return 0


if __name__ == "__main__":
    submission_strings = ["comments"]

    with open("./data/reproduction/red_flag_data.json", "r") as inputfile:
        id_data = json.load(inputfile)
    id_data_set = set(id_data)

    # with open("filter_results_percentiles_personalfinance.json", "r") as inputfile:
    #     pf_ids = json.load(inputfile)
    # with open("filter_results_percentiles_wallstreetbets.json", "r") as inputfile:
    #     wsb_ids = json.load(inputfile)
    # with open("filter_results_percentiles_explainlikeimfive.json", "r") as inputfile:
    #     elif_ids = json.load(inputfile)

    # special_filter_ids_set = set(pf_ids + wsb_ids + elif_ids)
    # print(len(special_filter_ids_set))
    for subreddit in ["personalfinance", "wallstreetbets"]:
        for submission_string in submission_strings:
            # print(f"Starting preprocessing of subreddit {subreddit} {submission_string}")
            file_name = f"{subreddit}_{submission_string}.zst"
            path = os.path.join("./data/subreddits_base_data", file_name)
            preprocessing(file_name, path)
            # current_length = get_amount_of_lines(os.path.join(PathManager.get_new_data_path(), file_name))
            # print(f"Subreddit {subreddit} has {current_length} {submission_string}")
            # print(f"Finished preprocessing of subreddit {subreddit} {submission_string}")