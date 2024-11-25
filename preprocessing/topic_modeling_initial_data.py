import json
import nltk
import os
import pandas as pd
import time
from bertopic import BERTopic
from datetime import datetime
from tqdm import tqdm
from umap import UMAP

from config.constants import SUBREDDITS
from pushshift import read_lines_zst
from utils.attribute_retrieval import get_basic_attribute

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()


def filter_title(obj: dict) -> bool:
    title = get_basic_attribute(obj, "title")
    if title is None or title == "[removed]" or title == "[deleted]":
        return True
    return False


def get_data(file_path: str) -> None:
    file_size = os.stat(file_path).st_size
    file_lines = 0
    created = None
    bad_lines = 0
    results = []
    for line, file_bytes_processed in read_lines_zst(file_path):
        file_lines += 1
        if file_lines % 100000 == 0:
            print(
                f"{created} Line: {file_lines:,} Bad Lines: {bad_lines:,} Bytes Processed: {file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(
                int(obj["created_utc"])).strftime("%Y/%m/%d")
            filter = filter_title(obj)
            if not filter:
                results.append({
                    "title": obj["title"],
                    "id": obj["id"]
                })

        except (KeyError, json.JSONDecodeError) as err:
            print("Error:" + err)
    results_df = pd.DataFrame(results)
    return results_df


def topic_modeling_initial_data() -> None:
    stopwords = nltk.corpus.stopwords.words("english")
    print(
        f'There are {len(stopwords)} default stopwords. They are {stopwords}')
    finished_length = 0

    for subreddit in tqdm(SUBREDDITS):
        data = get_data(f"./data/subreddits/{subreddit}_submissions.zst")
        print(f"Subreddit has {data.shape[0]} examples")
        data['text_without_stopwords'] = data['title'].apply(
            lambda x: ' '.join([w for w in x.split() if w.lower() not in stopwords]))
        # Lemmatization
        data['text_lemmatized'] = data['text_without_stopwords'].apply(
            lambda x: ' '.join([wn.lemmatize(w) for w in x.split() if w not in stopwords]))
        # Take a look at the data
        print(f"Handling subreddit {subreddit}")
        # Initiate UMAP
        umap_model = UMAP(n_neighbors=15,
                          n_components=5,
                          min_dist=0.0,
                          metric='cosine',
                          random_state=100)
        # Initiate BERTopic
        topic_model = BERTopic(umap_model=umap_model, language="english",
                               calculate_probabilities=True, verbose=True)
        # Run BERTopic model
        start = time.time()
        topics, probabilities = topic_model.fit_transform(
            data['text_lemmatized'])

        # Save the topic model
        topic_model.save(
            f"./data/explorative_analysis/topic_models_initial_data/{subreddit}")
        finished_length += data.shape[0]
        end = time.time()
        print(f"Iteration took {(end - start) / 60} minutes.")


if __name__ == "__main__":
    topic_modeling_initial_data()
