import argparse
import pandas as pd
import spacy
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from config.constants import SUBREDDITS


def characteristic_words(dataset_name: str) -> None:
    nlp = spacy.load('en_core_web_sm')
    data = load_dataset(dataset_name)
    train = data["train"]
    test = data["test"]
    dataset = train.concatenate(test)
    result = {}
    for el in tqdm(dataset):
        cur_subreddit = el["subreddit"]
        cur_text = el["text"]
        if cur_subreddit not in result:
            result[cur_subreddit] = cur_text
        else:
            result[cur_subreddit] = result[cur_subreddit] + " " + cur_text
    all_titles = []
    for subreddit in SUBREDDITS:
        all_titles.append(result[subreddit])
    df = pd.DataFrame({"subreddits": SUBREDDITS, "texts": all_titles})
    df.set_index("subreddits", inplace=True)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_separate = tfidf_vectorizer.fit_transform(df["texts"])

    df_tfidf = pd.DataFrame(
        tfidf_separate.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=df.index
    )
    columns = df_tfidf.columns.tolist()
    columns_combined = " ".join(columns)
    columns_spacy = nlp(columns_combined)
    new_columns = [token for token in columns_spacy if (
        token.pos_ == "NOUN" and len(token) > 3)]  # or token.pos_ == "VERB"]
    overall_columns = [str(token)
                       for token in new_columns if str(token) in columns]
    new_df_tfidf = df_tfidf[overall_columns].copy()
    new_df_tfidf.to_csv(
        "./data/explorative_analysis/tfidf_scores_per_subreddit_overall_filtered.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("gpt_data_formatting")
    parser.add_argument("-d", "--dataset_name", help="Data used for the inference",
                        type=str, default="Kris-Fillip/SocialFinanceQA")
    args = parser.parse_args()
    characteristic_words(dataset_name=args.dataset_name)
