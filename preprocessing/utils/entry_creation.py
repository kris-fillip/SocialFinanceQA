import sys

from config.constants import SUBMISSION_ATTRIBUTES_TO_COLLECT, COMMENT_ATTRIBUTES_TO_COLLECT
from utils.attribute_retrieval import get_basic_attribute
from utils.text_preprocessing import preprocess_text


def collect_parent_information(obj: dict, parent_ids_kept: dict, comment_is_submission: bool =False) -> dict:
    parent_obj = {}
    for attribute in SUBMISSION_ATTRIBUTES_TO_COLLECT:
        current_attribute_value = get_basic_attribute(obj, attribute)
        parent_obj[attribute] = current_attribute_value
    current_id = get_basic_attribute(obj, "name")
    if current_id is None:
        # id is only the unique identifier, the prefix t3_ signifies it being a submission
        # (needed to match to link_id/parent_id in comments)
        if comment_is_submission:
            current_id = "t1_" + obj["id"]
        else:
            current_id = "t3_" + obj["id"]
    parent_obj["id"] = current_id

    if comment_is_submission:
        body = get_basic_attribute(obj, "body")
        parent_obj["selftext"] = body
        parent_obj["title"] = ""
        # arbitrary high number instead of None since we don't know the actual amount of subcomments in advance
        parent_obj["num_comments"] = sys.maxsize

    parent_obj["comment_is_submission"] = comment_is_submission
    parent_ids_kept[current_id] = parent_obj
    return parent_ids_kept


def collect_comment_information(obj: dict) -> dict:
    comment_obj = {}
    for attribute in COMMENT_ATTRIBUTES_TO_COLLECT:
        current_attribute_value = get_basic_attribute(obj, attribute)
        comment_obj[attribute] = current_attribute_value
    return comment_obj


def collect_merged_information_two_answers(obj: dict, comment1: dict, comment2: dict, subreddit: str) -> dict:
    text = get_basic_attribute(obj, "title")
    if text is None:
        text = get_basic_attribute(obj, "body")
    context = get_basic_attribute(obj, "selftext")
    answer_1 = get_basic_attribute(comment1, "body")
    answer_2 = get_basic_attribute(comment2, "body")
    text = preprocess_text(text)
    context = preprocess_text(context)
    answer_1 = preprocess_text(answer_1)
    answer_2 = preprocess_text(answer_2)
    final_obj = {}
    final_obj["text"] = text
    final_obj["context"] = context
    final_obj["answer_1"] = answer_1
    final_obj["answer_2"] = answer_2
    final_obj["subreddit"] = subreddit

    for key, value in obj.items():
        if key not in ["title", "selftext", "comments"]:
            final_obj[key] = value
    for key, value in comment1.items():
        if key not in ["body", "parent_id"]:
            final_obj[key + "_answer1"] = value
    for key, value in comment2.items():
        if key not in ["body", "parent_id"]:
            final_obj[key + "_answer2"] = value

    return final_obj
