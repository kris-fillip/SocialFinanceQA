import argparse
import json
import os
from datetime import datetime

from config.constants import SUBREDDITS, SUBMISSION_STRING, COMMENT_STRING
from pushshift import read_lines_zst
from utils.attribute_retrieval import get_attribute, get_basic_attribute


def get_amount_of_lines(file_path: str) -> int:
    return len(list(read_lines_zst(file_path)))


def unique_attributes(obj: dict, result: dict) -> dict:
    # Collects unique attributes of a subreddit
    if "unique_attributes" not in result:
        result["unique_attributes"] = []
    for key in obj.keys():
        if key not in result["unique_attributes"]:
            result["unique_attributes"].append(key)
    return result


def unique_attribute_values(obj: dict, result: dict, first_n_results: int = 50) -> dict:
    for attr, value in obj.items():
        if attr not in result:
            result[attr] = [value]
        elif value not in result[attr] and len(result[attr]) < first_n_results:
            result[attr].append(value)
    return result


def count_unique_occurrences_of_attribute_per_day(obj: dict, result: dict, attribute: str) -> dict:
    created = datetime.utcfromtimestamp(
        int(obj["created_utc"])).strftime("%Y/%m/%d")
    if created not in result:
        result[created] = {}
    current_value = get_attribute(obj, attribute)
    if current_value is None:
        current_value = "unknown"
    if current_value not in result[created]:
        result[created][current_value] = 1
    else:
        result[created][current_value] += 1
    return result


def count_unique_occurrences_of_attribute(obj: dict, result: dict, attribute: str) -> dict:
    current_value = get_attribute(obj, attribute)
    if current_value is None:
        current_value = "unknown"
    if current_value not in result:
        result[current_value] = 1
    else:
        result[current_value] += 1
    return result


def collect_all_occurrences_of_numeric_attribute(obj: dict, result: dict, attribute: str) -> dict:
    if attribute not in result:
        result[attribute] = []
    current_value = get_basic_attribute(obj, attribute)
    result[attribute].append(current_value)
    return result


def archived(obj: dict, result: dict) -> dict:
    # this is a boolean indicating if a submission is archived
    # (e.g. after 180 days there is no voting/commenting possible anymore)
    return count_unique_occurrences_of_attribute_per_day(obj, result, "archived")


def authors_unique(obj: dict, result: dict) -> dict:
    # count unique occurrences of authors
    return count_unique_occurrences_of_attribute(obj, result, "author")


def author_flair_richtext_unique(obj: dict, result: dict) -> dict:
    # Count occurrences of unique author flair richtexts
    # These can be Emeritus Moderator, Subreddit Creator, or an empty list
    return count_unique_occurrences_of_attribute(obj, result, "author_flair_richtext")


def author_flair_text_unique(obj: dict, result: dict) -> dict:
    # Count occurrences of unique author flair texts
    # These can be Emeritus Moderator, Subreddit Creator, Wiki Contributor,
    # specific countries or even individual (real) names of authors
    return count_unique_occurrences_of_attribute(obj, result, "author_flair_text")


def domain_unique(obj: dict, result: dict) -> dict:
    # count occurrences of unique domains
    return count_unique_occurrences_of_attribute(obj, result, "domain")


def gilded_unique(obj: dict, result: dict) -> dict:
    # count occurrences of unique amounts of premium awards given to submissions
    return count_unique_occurrences_of_attribute(obj, result, "gilded")


def num_comments_unique(obj: dict, result: dict) -> dict:
    # Includes amount of comments a submissions received
    # TODO: Investigate negative amounts of comments (e.g. -1)
    return count_unique_occurrences_of_attribute(obj, result, "num_comments")


def link_flair_richtext(obj: dict, result: dict) -> dict:
    # These include general topics like "Credit", "Debt", "Investing", "Retirement", "Taxes", "Housing", "Insurance",
    return count_unique_occurrences_of_attribute_per_day(obj, result, "link_flair_richtext")


def link_flair_text(obj: dict, result: dict) -> dict:
    # These include general topics like "Credit", "Debt", "Investing", "Retirement", "Taxes", "Housing", "Insurance",
    # TODO: Check out custom flairs from users and their relevance (for now filter them out)
    return count_unique_occurrences_of_attribute_per_day(obj, result, "link_flair_text")


def post_retrieval_delta(obj: dict, result: dict) -> dict:
    # Keeping datetime.utcfromtimestamp for now to enable reproducibility
    if "created_utc" not in obj:
        return result
    created = datetime.utcfromtimestamp(
        int(obj["created_utc"])).strftime("%Y/%m/%d")
    if "retrieved_on" not in obj:
        if "retrieved_utc" not in obj:
            result[created] = -1
            return result
        retrieved = datetime.utcfromtimestamp(
            int(obj["retrieved_utc"])).strftime("%Y/%m/%d")
    else:
        retrieved = datetime.utcfromtimestamp(
            int(obj["retrieved_on"])).strftime("%Y/%m/%d")
    delta = (datetime.strptime(retrieved, "%Y/%m/%d") -
             datetime.strptime(created, "%Y/%m/%d")).days
    if created not in result or result[created] == -1:
        result[created] = delta
    return result


def iterative_average(avg: float, count: int, new_value: int | float) -> int | float:
    if count == 0:
        return new_value
    else:
        return (avg * count + new_value) / (count + 1)


def scoring(obj: dict, result: dict) -> dict:
    # Collects the score of a submission
    # TODO: Reverse Engineer ups and downs for submissions with upvote_ratio and score
    created = datetime.utcfromtimestamp(
        int(obj["created_utc"])).strftime("%Y/%m/%d")
    if created not in result:
        result[created] = {'score': 0, 'ups': 0, 'downs': 0, 'upvote_ratio': 0, 'count': 0, 'score_avg': 0,
                           'ups_avg': 0, 'downs_avg': 0, 'upvote_ratio_avg': 0.5}
    score_handle = {
        'score': 0,
        'ups': 0,
        'downs': 0,
        'upvote_ratio': 0.5,
    }
    for attribute, value in score_handle.items():
        current_attribute_value = get_basic_attribute(obj, attribute)
        if current_attribute_value is not None:
            result[created][attribute] += 1
            result[created][attribute + "_avg"] = iterative_average(result[created][attribute + "_avg"],
                                                                    result[created]['count'], current_attribute_value)
        else:
            result[created][attribute + "_avg"] = iterative_average(result[created][attribute + "_avg"],
                                                                    result[created]['count'], value)
    result[created]['count'] += 1
    return result


def score_occurrences(obj: dict, result: dict) -> dict:
    return collect_all_occurrences_of_numeric_attribute(obj, result, "score")


def upvote_ratio_occurrences(obj: dict, result: dict) -> dict:
    return collect_all_occurrences_of_numeric_attribute(obj, result, "upvote_ratio")


def num_comments_occurrences(obj: dict, result: dict) -> dict:
    return collect_all_occurrences_of_numeric_attribute(obj, result, "num_comments")


def total_awards_received_occurrences(obj: dict, result: dict) -> dict:
    return collect_all_occurrences_of_numeric_attribute(obj, result, "total_awards_received")


def title_occurrences(obj: dict, result: dict) -> dict:
    return collect_all_occurrences_of_numeric_attribute(obj, result, "title")


def selftext_occurrences(obj: dict, result: dict) -> dict:
    return collect_all_occurrences_of_numeric_attribute(obj, result, "selftext")


def body_occurrences(obj: dict, result: dict) -> dict:
    return collect_all_occurrences_of_numeric_attribute(obj, result, "body")


def gilded_occurrences(obj: dict, result: dict) -> dict:
    return collect_all_occurrences_of_numeric_attribute(obj, result, "gilded")


def subreddit_subscribers(obj: dict, result: dict) -> dict:
    # Only the amount of subscribers of a subreddit at the retrieval date
    if "retrieved_on" not in obj:
        if "retrieved_utc" not in obj:
            return result
        retrieved = datetime.utcfromtimestamp(
            int(obj["retrieved_utc"])).strftime("%Y/%m/%d")
    else:
        retrieved = datetime.utcfromtimestamp(
            int(obj["retrieved_on"])).strftime("%Y/%m/%d")
    if "subreddit_subscribers" not in obj:
        return result
    if retrieved not in result or obj["subreddit_subscribers"] > result[retrieved]:
        result[retrieved] = obj["subreddit_subscribers"]
    return result


def stickied(obj: dict, result: dict, select_true: bool = True) -> dict:
    # select submissions which are stickied to the top of the subreddit
    if "stickied" not in result:
        result["stickied"] = []
    if "stickied" in obj and obj["stickied"] == select_true:
        cur_result = {
            "created_utc": obj["created_utc"],
            "author": obj["author"],
            "title": obj["title"],
            "selftext": obj["selftext"]}
        result["stickied"].append(cur_result)
    return result


def iterate_over_file(file_path: str, analytical_functions: list, subreddit: str, submission: bool = True) -> None:
    file_size = os.stat(file_path).st_size
    file_lines = 0
    created = None
    bad_lines = 0
    results = {}
    for line, file_bytes_processed in read_lines_zst(file_path):
        file_lines += 1
        if file_lines % 100000 == 0:
            print(
                f"{created} Line: {file_lines:,} Bad Lines: {bad_lines:,} Bytes Processed: {file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(
                int(obj["created_utc"])).strftime("%Y/%m/%d")
            # TODO: investigate impact of embedded media, num_crossposts
            for function in analytical_functions:
                function_name = function.__name__
                if function_name not in results:
                    results[function_name] = {}
                results[function_name] = function(obj, results[function_name])

        except (KeyError, json.JSONDecodeError) as err:
            print("Error:" + err)

    if submission:
        data_type = SUBMISSION_STRING
    else:
        data_type = COMMENT_STRING
    # WRITE TO JSON
    for analytical_result in results:
        output_file_path = os.path.join("/data/analysis", f"{subreddit}",
                                        f"{data_type}_{analytical_result}.json")
        with open(output_file_path, "w") as file_handle:
            json.dump(results[analytical_result], file_handle)


def create_analysis_directories() -> None:
    for current_subreddit in SUBREDDITS:
        cur_path = os.path.join("/data/analysis", f"{current_subreddit}")
        exists = os.path.exists(cur_path)
        if not exists:
            # Create a new directory because it does not exist
            os.makedirs(cur_path)
            print("created directory: " + cur_path)


def analysis(complete: bool) -> None:
    if complete:
        # run complete set of analytical functions to get insights into the relevant attributes of the overall datasets
        analytical_functions_submissions = [unique_attributes, unique_attribute_values, archived, authors_unique,
                                            author_flair_richtext_unique, author_flair_text_unique, domain_unique,
                                            gilded_unique, link_flair_richtext, link_flair_text, num_comments_unique,
                                            post_retrieval_delta, scoring, score_occurrences, upvote_ratio_occurrences,
                                            subreddit_subscribers, stickied, num_comments_occurrences, gilded_occurrences,
                                            total_awards_received_occurrences, title_occurrences, selftext_occurrences]
        analytical_functions_comments = [unique_attributes, unique_attribute_values, score_occurrences,
                                         upvote_ratio_occurrences, body_occurrences, gilded_occurrences,
                                         total_awards_received_occurrences]
    else:
        # run only analytical functions relevant for percentiles filtering
        analytical_functions_submissions = [
            score_occurrences, upvote_ratio_occurrences]
        analytical_functions_comments = [
            score_occurrences, gilded_occurrences, total_awards_received_occurrences]
    create_analysis_directories()
    for subreddit in SUBREDDITS:
        file_name = f"{subreddit}_{SUBMISSION_STRING}.zst"
        path = os.path.join("/data/subreddits", file_name)
        iterate_over_file(path, analytical_functions_submissions,
                          subreddit, submission=True)
        file_name = f"{subreddit}_{COMMENT_STRING}.zst"
        path = os.path.join("/data/subreddits", file_name)
        iterate_over_file(path, analytical_functions_comments,
                          subreddit, submission=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("analysis")
    parser.add_argument("-c", "--complete", help="Boolean indicating if original dataset should be reproducted",
                        default=True, action="store_false")
    args = parser.parse_args()
    analysis(complete=args.complete)
