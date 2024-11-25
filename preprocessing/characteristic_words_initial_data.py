import json
import os
import pandas as pd
import spacy
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

from config.constants import SUBREDDITS
from pushshift import read_lines_zst
from utils.attribute_retrieval import get_basic_attribute


def preprocessing(current_path, nlp):
    file_size = os.stat(current_path).st_size
    file_lines = 0
    created = None
    bad_lines = 0
    combined_titles = ""
    for line, file_bytes_processed in read_lines_zst(current_path):
        file_lines += 1
        if file_lines % 1000 == 0:
            print(
                f"{created} Line: {file_lines:,} Bad Lines: {bad_lines:,} Bytes Processed: {file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(
                int(obj["created_utc"])).strftime("%Y/%m/%d")
            new_title = get_basic_attribute(obj, "title").lower()
            spacy_title = nlp(new_title)
            lemmatized_tokens = [token.lemma_ for token in spacy_title]
            lemmatized_text = ' '.join(lemmatized_tokens)
            if combined_titles == "":
                combined_titles = lemmatized_text
            else:
                combined_titles += " " + lemmatized_text

        except (KeyError, json.JSONDecodeError) as err:
            print("Error:" + err)
            bad_lines += 1
    print("Got subreddit title combination!")
    return combined_titles


def get_amount_of_lines(file_path):
    return len(list(read_lines_zst(file_path)))


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def characteristic_words_initial_data():
    all_titles = []
    nlp = spacy.load('en_core_web_sm')
    overall_lines = 0
    for subreddit in SUBREDDITS:
        print(f"Starting reading length of subreddit {subreddit}")
        file_name = f"{subreddit}_submissions.zst"
        path = os.path.join("./data/subreddits", file_name)
        cur_lines = get_amount_of_lines(path)
        print(f"Subreddit has {cur_lines} lines.")
        overall_lines += cur_lines
    print(f"Overall lines to evaluate: {overall_lines}")
    for subreddit in SUBREDDITS:
        print(f"Starting preprocessing of subreddit {subreddit}")
        file_name = f"{subreddit}_submissions.zst"
        path = os.path.join("./data/subreddits", file_name)
        all_titles.append(preprocessing(path, nlp))
    df = pd.DataFrame({"subreddits": SUBREDDITS, "texts": all_titles})
    df.set_index("subreddits", inplace=True)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_separate = tfidf_vectorizer.fit_transform(df["texts"])

    df_tfidf = pd.DataFrame(
        tfidf_separate.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=df.index
    )
    columns = df_tfidf.columns.tolist()
    columns_chunked = list(divide_chunks(columns, 100000))
    overall_columns = []
    for cur_columns in columns_chunked:
        tokenized = nlp(" ".join(cur_columns))
        for token in tokenized:
            if (token.pos_ == "NOUN" and len(str(token)) > 3 and str(token) in columns):
                overall_columns.append(str(token))
    new_df_tfidf = df_tfidf[overall_columns].copy()
    new_df_tfidf.to_csv(
        "./data/explorative_analysis/tfidf_scores_per_subreddit_overall.csv")


if __name__ == "__main__":
    characteristic_words_initial_data()
