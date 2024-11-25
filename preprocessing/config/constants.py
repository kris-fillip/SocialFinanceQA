import re


PERCENTILES = ["90"]

SUBREDDITS = ["personalfinance", "financialindependence", "FinancialPlanning", "investing", "wallstreetbets", "Wallstreetbetsnew",
              "stocks", "StockMarket", "pennystocks", "options", "RealEstate", "Economics", "realestateinvesting", "AskEconomics", "explainlikeimfive"]

SUBMISSION_STRING = "submissions"

COMMENT_STRING = "comments"

COMMENTS_PERCENTILE_ATTRIBUTES = ["score", "upvote_ratio", "num_comments"]

SUBMISSIONS_PERCENTILE_ATTRIBUTES = [
    "score", "upvote_ratio", "body", "gilded", "total_awards_received"]

GILDING_AND_AWARDINGS = "gildings_and_awardings"

NO_GILDING_AND_AWARDINGS = "no_gilding_and_awards"

CONSIDER_GILDING_AND_AWARDING = [NO_GILDING_AND_AWARDINGS]

MIN_SCORE = 3

MIN_THRESHOLD = 0.75

MIN_AWARDS = 2

REDDIT_USER = r"(?:\/?u\/\w+)"

REDDIT_USER_RE = re.compile(REDDIT_USER, flags=re.UNICODE)

HASH_RE = re.compile(r'#(?=\w+)', re.UNICODE)

URL_RE = re.compile(
    r"""((https?:\/\/|www)|\w+\.(\w{2-3}))([\w\!#$&-;=\?\-\[\]~]|%[0-9a-fA-F]{2})+""", re.UNICODE)

MAX_LEVEL = 0

FILTER_ELEMENTS = ["Edit: ", "edit: ", "update: ", "Update: "]

RICHTEXT_ATTRIBUTES = ["author_flair_richtext", "link_flair_richtext"]

SUBMISSION_ATTRIBUTES_TO_COLLECT = ["num_comments", "title", "selftext", "score",
                                    "upvote_ratio", "ups", "downs", "author", "created_utc", "retrieved_on", "retrieved_utc"]

COMMENT_ATTRIBUTES_TO_COLLECT = ["body", "score", "upvote_ratio", "ups", "downs",
                                 "author", "parent_id", "name", "id", "created_utc", "retrieved_on", "retrieved_utc"]

QUESTION_HIT_WORDS = ["help me", "need help", "any help", "please help", "get some help", "please advise",
                      "advice", "recommendations", "can i ", "should i ", "do i ", "anyone know", "does it make sense"]

ALPHABETS = "([A-Za-z])"

PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"

SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"

STARTERS = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"

ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"

WEBSITES = "[.](com|net|org|io|gov|edu|me)"

DIGITS = "([0-9])"

MULTIPLE_DOTS = r'\.{2,}'

MAX_TOKENIZED_LENGTH = 1023

TEST_SIZE = 500

MIN_BODY_WORDS = 30
