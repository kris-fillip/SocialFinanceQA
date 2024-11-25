from config.constants import MIN_BODY_WORDS
from utils.attribute_retrieval import get_basic_attribute, get_richtext
from utils.question_detection import evaluate_whether_question


def filter_score(obj: dict, perc: float, min_score: int = 2) -> bool:
    score = get_basic_attribute(obj, "score")
    if score is None or score < min_score or score < perc:
        return True
    return False


def filter_upvote_ratio(obj: dict, perc: float, threshold: float = 0.5) -> bool:
    upvote_ratio = get_basic_attribute(obj, "upvote_ratio")
    if upvote_ratio is not None:
        if upvote_ratio < threshold or upvote_ratio < perc:
            return True
    return False


def filter_num_comments(obj: dict, min_comments: int = 3) -> bool:
    num_comment = get_basic_attribute(obj, "num_comments")
    if num_comment is not None:
        if num_comment < min_comments:
            return True
    return False


def filter_domain(obj: dict) -> bool:
    domain = get_basic_attribute(obj, "domain")
    if domain and "self." not in domain:
        return True
    return False


def filter_link_flair_text(obj: dict, subreddit: str) -> bool:
    # only potentially relevant if trying to filter for questions/content on specific topic
    link_flair = get_basic_attribute(obj, "link_flair_text")
    if subreddit == "AskEconomics":
        askeconomics_valid_flairs = [
            "Approved Answers", "Good Question", "Simple Questions/Career"]
        if link_flair not in askeconomics_valid_flairs:
            return True
    if subreddit == "financialindependence":
        f_i_invalid_flairs = ["Mod Post", "Case Study",
                              "Moderator Meta", "Personal Journey"]
        if link_flair in f_i_invalid_flairs:
            return True
    if subreddit == "explainlikeimfive":
        if link_flair != "Economics":
            return True
    return False


def filter_author_flair(obj: dict) -> bool:
    author_flair_richtext = get_richtext(obj, "author_flair_richtext")
    if author_flair_richtext == "Emeritus Moderator":
        return True
    return False


def filter_bot_author(obj: dict) -> bool:
    author_flair = get_basic_attribute(obj, "author_flair_text")
    author = get_basic_attribute(obj, "author")
    bot_flairs = ["IndexBot", "AutoModerator", "Moderation Bot"]
    if author_flair is not None and author_flair in bot_flairs:
        return True
    if author is not None and author in bot_flairs:
        return True
    return False


def filter_content(obj: dict, comment_as_submission: bool = False) -> bool:
    if comment_as_submission == True:
        body = get_basic_attribute(obj, "body")
        if body is None or body == "" or body == "[removed]" or body == "[deleted]":
            return True
    else:
        selftext = get_basic_attribute(obj, "selftext")
        title = get_basic_attribute(obj, "title")

        if selftext is None or title is None or (selftext == "" and title == ""):
            return True
        elif selftext == "[removed]" or selftext == "[deleted]":
            return True
        elif title == "[removed]" or title == "[deleted]":
            return True
    return False


def filter_body(obj: dict) -> bool:
    body = get_basic_attribute(obj, "body")
    if body is None or body == "" or body == "[removed]" or body == "[deleted]" or len(body.split()) < MIN_BODY_WORDS:
        return True
    return False


def filter_stickied(obj: dict) -> bool:
    stickied = get_basic_attribute(obj, "stickied")
    if stickied:
        return True
    return False


def filter_collapsed(obj: dict) -> bool:
    collapsed = get_basic_attribute(obj, "collapsed")
    if collapsed:
        return True
    return False


def filter_gilded(obj: dict, perc: float, min_gildings: int = 1) -> bool:
    gilded = get_basic_attribute(obj, "gilded")
    if gilded is not None:
        if gilded < min_gildings or gilded < perc:
            return True
    return False


def filter_awardings(obj: dict, perc: float, min_awards: int = 2) -> bool:
    awards_received = get_basic_attribute(obj, "total_awards_received")
    if awards_received is not None:
        if awards_received < min_awards or awards_received < perc:
            return True
    all_awardings = get_basic_attribute(obj, "all_awardings")
    if all_awardings is not None and len(all_awardings) > 0:
        if len(all_awardings) == 1:
            if all_awardings[0]["count"] < 2:
                return True
        elif all_awardings[0]["count"] < 2 and all_awardings[1]["count"] < 2:
            return True
    return False


def filter_distinguished(obj: dict, subreddit: str) -> bool:
    if subreddit not in ["AskEconomics", "financialindependence", "StockMarket", "options", "RealEstate", "Economics"]:
        return False
    distinguished = get_basic_attribute(obj, "distinguished")
    if distinguished == "moderator" or distinguished == "admin":
        return True
    return False


def filter_question(obj: dict, attributes: list[str], subreddit: str, good_question_ids: bool = None) -> bool:
    if subreddit == "explainlikeimfive":
        return False
    cur_id = "t3_" + get_basic_attribute(obj, "id")

    if good_question_ids and subreddit in good_question_ids and cur_id in good_question_ids[subreddit]:
        return False
    for attribute in attributes:
        if evaluate_whether_question(obj, attribute):
            return False
    return True
