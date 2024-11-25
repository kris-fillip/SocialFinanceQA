import copy
import json
import os
import zstandard
from datetime import datetime

from attribute_filters import *
from config.constants import PERCENTILES, SUBMISSION_STRING, COMMENT_STRING, SUBREDDITS, GILDING_AND_AWARDINGS, MIN_SCORE, MIN_AWARDS, MIN_THRESHOLD, CONSIDER_GILDING_AND_AWARDING
from percentiles import get_percentiles
from pushshift import read_lines_zst, write_line_zst
from utils.attribute_retrieval import get_basic_attribute
from utils.entry_creation import collect_comment_information, collect_parent_information


def filter_object(obj: dict, subreddit: str, submission_string: str, curated_or_not: str, percentiles_dict: dict, percentile: int, good_question_ids: dict, comment_as_submission: bool = False) -> bool:
    if comment_as_submission == True:
        if filter_score(obj, perc=percentiles_dict["score"], min_score=MIN_SCORE):
            return True, "Score too low"
        if filter_content(obj, comment_as_submission=True):
            return True, "part of content is [deleted] or [removed] or completely empty"
        if filter_author_flair(obj):
            return True, "Author flair is Moderator/Creator"
        if filter_bot_author(obj):
            return True, "Submission was written by bot"
        if filter_question(obj, ["body"], subreddit, good_question_ids):
            return True, "Submission does not contain a question"

    elif submission_string == "submissions":
        if percentile != 0:
            if filter_score(obj, perc=percentiles_dict["score"], min_score=MIN_SCORE):
                return True, "Score too low"
            if filter_upvote_ratio(obj, perc=percentiles_dict["upvote_ratio"], threshold=MIN_THRESHOLD):
                return True, "Upvote ratio too low"
        if filter_num_comments(obj, min_comments=3):
            return True, "Number of comments too low"
        if filter_domain(obj):
            return True, "Domain is self"
        if filter_author_flair(obj):
            return True, "Author flair is Moderator/Creator"
        if filter_content(obj):
            return True, "part of content is [deleted] or [removed] or completely empty"
        if filter_question(obj, ["title", "selftext"], subreddit, good_question_ids):
            return True, "Submission does not contain a question"
        if filter_stickied(obj):
            return True, "Stickied"
        if filter_bot_author(obj):
            return True, "Submission was written by bot"
        if filter_link_flair_text(obj, subreddit):
            return True, "Submission has irrelevant link flair"
        if filter_distinguished(obj, subreddit):
            return True, "Submission is likely mod content"

    elif submission_string == "comments":
        if filter_body(obj):
            return True, "Body is [deleted] or [removed] or too short"
        if filter_collapsed(obj):
            return True, "Comment is collapsed"
        if filter_bot_author(obj):
            return True, "Comment was written by bot"
        if curated_or_not == GILDING_AND_AWARDINGS:
            if filter_gilded(obj, percentiles_dict["gilded"], min_gildings=2):
                return True, "Comment wasn't gilded enough"
            if filter_awardings(obj, percentiles_dict["total_awards_received"], min_awards=MIN_AWARDS):
                return True, "Comment wasn't awarded enough"

    return False, "Not filtered"


def iterate_over_file(file_path: str, stats: dict, curated_or_not: str, percentiles_dict: dict, percentile: int, subreddit: str, submission_string: str, comment_as_submission_ids: dict, good_question_ids: dict, submission: bool = True) -> None:
    file_size = os.stat(file_path).st_size
    file_lines = 0
    created = None
    bad_lines = 0
    filtered = 0
    # Output for submissions and comments which are kept
    output_path = f"./data/filtered/{curated_or_not}_percentile{percentile}/{subreddit}_filtered_{submission_string}.zst"
    handle = zstandard.ZstdCompressor().stream_writer(open(output_path, 'wb'))
    parent_ids_kept = {}
    if not submission:
        # Output for combination of submissions and comments which are kept
        output_path_combined = f"./data/filtered/{curated_or_not}_percentile{percentile}/{subreddit}_filtered_combined.zst"
        handle_combined = zstandard.ZstdCompressor().stream_writer(
            open(output_path_combined, 'wb'))
        parents = json.load(open(
            f"./data/filtered/{curated_or_not}_percentile{percentile}/{subreddit}_parent_ids_kept.json"))
    for line, file_bytes_processed in read_lines_zst(file_path):
        file_lines += 1
        if file_lines % 100000 == 0:
            print(
                f"{created} Line: {file_lines:,} Bad Lines: {bad_lines:,} Bytes Processed: {file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(
                int(obj["created_utc"])).strftime("%Y/%m/%d")

            if not submission:
                # Additional Handling for comments_as_submissions
                current_link_id = obj["link_id"]
                current_parent_id = obj["parent_id"]
                if subreddit in comment_as_submission_ids and current_parent_id in comment_as_submission_ids[subreddit]:
                    filter_obj, _ = filter_object(obj, subreddit, submission_string, curated_or_not,
                                                  percentiles_dict, percentile, good_question_ids, comment_as_submission=True)
                    if filter_obj:
                        filtered += 1
                        continue
                    parents = collect_parent_information(
                        obj, parents, comment_is_submission=True)
                else:
                    # Filter comments of submissions which are already filtered
                    if current_link_id in parents:
                        current_id = current_link_id
                    elif current_parent_id in parents:
                        current_id = current_parent_id
                    else:
                        filtered += 1
                        continue
                    parents[current_id]['num_comments'] -= 1
                    if 'comments' not in parents[current_id]:
                        parents[current_id]['comments'] = []
                    filter_obj, _ = filter_object(
                        obj, subreddit, submission_string, curated_or_not, percentiles_dict, percentile, good_question_ids)
                    if filter_obj:
                        if parents[current_id]['num_comments'] == 0:
                            parents[current_id]['num_comments'] = len(
                                parents[current_id]['comments'])
                            write_line_zst(handle_combined,
                                           json.dumps(parents[current_id]))
                            del parents[current_id]
                        filtered += 1
                        continue
                    current_comment = collect_comment_information(obj)
                    parents[current_id]['comments'].append(current_comment)
                    if parents[current_id]['num_comments'] == 0:
                        parents[current_id]['num_comments'] = len(
                            parents[current_id]['comments'])
                        write_line_zst(handle_combined,
                                       json.dumps(parents[current_id]))
                        del parents[current_id]
            if submission:
                current_id = get_basic_attribute(obj, "name")
                if current_id is None:
                    current_id = "t3_" + obj["id"]
                # Filter submissions where comments are actual submissions/questions
                if subreddit in comment_as_submission_ids and current_id in comment_as_submission_ids[subreddit]:
                    filtered += 1
                    continue
                filter_obj, _ = filter_object(
                    obj, subreddit, submission_string, curated_or_not, percentiles_dict, percentile, good_question_ids)
                if filter_obj:
                    filtered += 1
                    continue
                parent_ids_kept = collect_parent_information(
                    obj, parent_ids_kept)
            write_line_zst(handle, line)

        except (KeyError, json.JSONDecodeError) as err:
            print("Error:" + err)
    handle.close()
    if subreddit not in stats:
        stats[subreddit] = {
            "filtered": filtered,
            "total": file_lines,
            "kept": file_lines - filtered,
            "percentage_kept": round((file_lines - filtered) / file_lines * 100, 2),
        }
    print(f"Filtered out {filtered} of {file_lines} {submission_string}")
    print(f"Wrote {file_lines - filtered} {submission_string} to {output_path}")
    if not submission:
        print(f"{len(parents)} submissions not matched with all their comments")
        temp_parents = copy.deepcopy(parents)
        for parent_id in temp_parents:
            if "comments" in temp_parents[parent_id] and len(temp_parents[parent_id]['comments']) > 0:
                temp_parents[parent_id]['num_comments'] = len(
                    temp_parents[parent_id]['comments'])
                write_line_zst(handle_combined, json.dumps(
                    temp_parents[parent_id]))
                del parents[parent_id]
        handle_combined.close()
        print(f"{len(parents)} submissions still not matched with comments")
        with open(f"./data/filtered/{curated_or_not}_percentile{percentile}/{subreddit}_parent_ids_kept_after_merging.json",
                  "w") as file_handle:
            json.dump(parents, file_handle)
    if submission:
        with open(f"./data/filtered/{curated_or_not}_percentile{percentile}/{subreddit}_parent_ids_kept.json",
                  "w") as file_handle:
            json.dump(parent_ids_kept, file_handle)


def create_filtering_directory(curated_or_not: str, percentile: int) -> None:
    cur_path = f"./data/filtered/{curated_or_not}_percentile{percentile}"
    os.makedirs(cur_path, exist_ok=True)
    print("created directory: " + cur_path)


def filtering() -> None:
    submission_stats = {}
    comment_stats = {}

    comment_as_submission_files = os.listdir(
        "./data/special_submission_ids/comment_is_submission")
    comment_as_submission_ids = {}
    for el in comment_as_submission_files:
        subreddit_name = el.split(".")[0]
        with open(os.path.join("./data/special_submission_ids/comment_is_submission", el), "r") as id_input_file:
            cur_ids = json.load(id_input_file)
        comment_as_submission_ids[subreddit_name] = cur_ids

    good_question_files = os.listdir(
        "./data/special_submission_ids/good_questions")
    good_question_ids = {}
    for el in good_question_files:
        subreddit_name = el.split(".")[0]
        with open(os.path.join("./data/special_submission_ids/good_questions", el), "r") as id_input_file:
            cur_ids = json.load(id_input_file)
        good_question_ids[subreddit_name] = cur_ids

    # For the pipeline to properly work submissions always have to be filtered in advance to their respective comments
    submission_or_comment = [True, False]
    percentiles_dict_overall = get_percentiles()
    for curated_or_not in CONSIDER_GILDING_AND_AWARDING:
        for percentile in PERCENTILES:
            create_filtering_directory(curated_or_not, percentile)
            print(f"Handling {curated_or_not}_percentile{percentile}")
            submission_stats = {}
            comment_stats = {}
            for subreddit in SUBREDDITS:
                print(f"Handling subreddit {subreddit}")
                for is_submission in submission_or_comment:
                    if is_submission:
                        submission_string = SUBMISSION_STRING
                    else:
                        submission_string = COMMENT_STRING
                    percentiles_dict = percentiles_dict_overall[percentile][subreddit][submission_string]
                    file_name = f"{subreddit}_{submission_string}.zst"
                    path = os.path.join("./data/subreddits", file_name)
                    if is_submission:
                        iterate_over_file(path, submission_stats, curated_or_not, percentiles_dict, percentile, subreddit,
                                          submission_string, comment_as_submission_ids, good_question_ids, submission=is_submission)
                    else:
                        iterate_over_file(path, comment_stats, curated_or_not, percentiles_dict, percentile, subreddit,
                                          submission_string, comment_as_submission_ids, good_question_ids, submission=is_submission)
            with open(f"./data/filtered/{curated_or_not}_percentile{percentile}/submissions_filtered_stats.json",
                      "w") as stats_handle:
                json.dump(submission_stats, stats_handle)
            with open(f"./data/filtered/{curated_or_not}_percentile{percentile}/comments_filtered_stats.json",
                      "w") as stats_handle:
                json.dump(comment_stats, stats_handle)


if __name__ == "__main__":
    filtering()
