import argparse
import json
import nltk
import pandas as pd
import time
from bertopic import BERTopic
from datasets import load_dataset
from tqdm import tqdm
from umap import UMAP

from config.constants import SUBREDDITS

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()


def topic_modeling(dataset_name: str) -> None:
    data = load_dataset(dataset_name)
    train = data["train"]
    test = data["test"]
    data = train.concatenate(test)

    df = pd.DataFrame(data)
    overall_length = df.shape[0]
    stopwords = nltk.corpus.stopwords.words("english")
    print(
        f'There are {len(stopwords)} default stopwords. They are {stopwords}')
    finished_length = 0
    topic_counts = []
    print(df["subreddit"].value_counts())
    for subreddit in tqdm(SUBREDDITS):
        cur_df = df[df["subreddit"] == subreddit].copy()
        cur_df['text_without_stopwords'] = cur_df['text'].apply(
            lambda x: ' '.join([w for w in x.split() if w.lower() not in stopwords]))
        # Lemmatization
        cur_df['text_lemmatized'] = cur_df['text_without_stopwords'].apply(
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
                               calculate_probabilities=True, verbose=True, nr_topics="auto")
        topic_model_no_merging = BERTopic(
            umap_model=umap_model, language="english", calculate_probabilities=True, verbose=True)
        # Run BERTopic model
        start = time.time()
        try:
            topics, probabilities = topic_model.fit_transform(
                cur_df['text_lemmatized'])
        except:
            print(f"No topic shrinkage for subreddit: {subreddit}")
            topics, probabilities = topic_model_no_merging.fit_transform(
                cur_df['text_lemmatized'])
            topic_model = topic_model_no_merging
        topic_results = topic_model.get_topic_info()
        topic_results.to_csv(
            f"./data/explorative_analysis/topic_models/topics_{subreddit}.csv", sep=";")
        try:
            fig = topic_model.visualize_barchart(top_n_topics=20, n_words=10)
            fig.write_image(
                f"./data/explorative_analysis/topic_models/barchart_{subreddit}.png")
        except (KeyError, json.JSONDecodeError, ValueError, IndexError) as err:
            print("Error:" + str(err))
        # try:
        #     fig = topic_model.visualize_topics(top_n_topics=20)
        #     fig.write_image(f"./topic_models/similarities_{subreddit}.png")
        # except (KeyError, json.JSONDecodeError, ValueError) as err:
        #     print("Error:" + str(err))
        try:
            fig = topic_model.visualize_hierarchy(top_n_topics=20)
            fig.write_image(
                f"./data/explorative_analysis/topic_models/hierarchy_{subreddit}.png")
        except (KeyError, json.JSONDecodeError, ValueError, IndexError) as err:
            print("Error:" + str(err))
        try:
            fig = topic_model.visualize_heatmap()
            fig.write_image(
                f"./data/explorative_analysis/topic_models/heatmap_{subreddit}.png")
        except (KeyError, json.JSONDecodeError, ValueError, IndexError) as err:
            print("Error:" + str(err))
        topic_prediction = topic_model.topics_[:]
        cur_df["topics"] = topic_prediction
        cur_topic_counts = {
            subreddit: cur_df["topics"].value_counts().to_dict()
        }
        topic_counts.append(cur_topic_counts)
        # Save the topic model
        topic_model.save(
            f"./data/explorative_analysis/topic_models/{subreddit}")
        finished_length += cur_df.shape[0]
        print(
            f"Finished {round(finished_length * 100 / overall_length, 2)} % of dataset")
        end = time.time()
        print(f"Iteration took {(end - start) / 60} minutes.")
    with open(f"./data/explorative_analysis/topic_models/topic_counts_per_subreddit.json", "w") as outputfile:
        json.dump(topic_counts, outputfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("topic_modeling")
    parser.add_argument("-d", "--dataset_name", help="Data used for the inference",
                        type=str, default="Kris-Fillip/SocialFinanceQA")
    args = parser.parse_args()
    topic_modeling(dataset_name=args.dataset_name)
