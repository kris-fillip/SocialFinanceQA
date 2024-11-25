import json
import os
import numpy as np

from config.constants import COMMENT_STRING, COMMENTS_PERCENTILE_ATTRIBUTES, PERCENTILES, SUBMISSION_STRING, SUBMISSIONS_PERCENTILE_ATTRIBUTES, SUBREDDITS


def get_percentiles() -> dict:
    # Calculate all required percentiles for submissions and comments of all considered subreddits
    if os.path.exists("./data/percentiles/percentiles.json"):
        with open("./data/percentiles/percentiles.json", "r") as percentiles_handle:
            percentiles_dict_overall = json.load(percentiles_handle)
        return percentiles_dict_overall
    percentiles_dict_overall = {}
    for percentile in PERCENTILES:
        if percentile not in percentiles_dict_overall:
            percentiles_dict_overall[percentile] = {}
        for subreddit in SUBREDDITS:
            if subreddit not in percentiles_dict_overall[percentile]:
                percentiles_dict_overall[percentile][subreddit] = {}
            for entity_name in [SUBMISSION_STRING, COMMENT_STRING]:
                if entity_name not in percentiles_dict_overall[percentile][subreddit]:
                    percentiles_dict_overall[percentile][subreddit][entity_name] = {}
                percentiles_dict_overall[percentile][subreddit][entity_name] = create_percentile(percentile, subreddit, entity_name)
                print(f"Added percentiles {percentile} for subreddit {subreddit} {entity_name}")
    with open("./data/percentiles/percentiles.json", "w") as percentiles_handle:
        json.dump(percentiles_dict_overall, percentiles_handle)
    return percentiles_dict_overall


def create_percentile(percentile: int, subreddit: str, entity_name: str) -> dict:
    percentile_scores = {}
    attributes = []
    if entity_name == SUBMISSION_STRING:
        attributes = SUBMISSIONS_PERCENTILE_ATTRIBUTES
    elif entity_name == COMMENT_STRING:
        attributes = COMMENTS_PERCENTILE_ATTRIBUTES
    percentile_filter = int(percentile)
    for attr in attributes:
        with open(f"./data/analysis/{subreddit}/{entity_name}_{attr}_occurrences.json", "r") as f:
            unique_attribute_values = json.load(f)
        if attr == "body":
            unique_attribute_values[attr] = [body for body in unique_attribute_values[attr] if body is not None]
            unique_attribute_values[attr] = [len(body.split()) for body in unique_attribute_values[attr]]
        ar = np.array(unique_attribute_values[attr])
        ar = ar[ar != np.array(None)]
        if len(ar) == 0:
            cur_percentile = 0.0
        else:
            cur_percentile = np.percentile(ar, percentile_filter)
        percentile_scores[attr] = cur_percentile
    return percentile_scores


if __name__ == "__main__":
    get_percentiles()