import json


def hatebert_filtering() -> None:
    with open("./data/hatebert_results/offense.json", "r") as input_file:
        offense = json.load(input_file)

    with open("./data/question_answers/no_gilding_and_awards_percentile90/complete_qa.json", "r") as input_file:
        data = json.load(input_file)

    filtered_ids = []
    non_filtered_ids = []
    for key, value in offense.items():
        idx = int(key)
        if value == "NOT":
            filtered_ids.append(data[idx]["id"])
        else:
            filtered_ids.append(data[idx]["id"])

    print(f"Filtered {len(filtered_ids)} entries")
    print(f"Kept {len(non_filtered_ids)} entries")

    non_filtered_ids_set = set(non_filtered_ids)
    filtered_ids_set = set(filtered_ids)
    subreddit_results = {}
    non_filtered_data = []
    for el in data:
        if el["id"] in filtered_ids_set:
            cur_subreddit = el["subreddit"]
            if cur_subreddit not in subreddit_results:
                subreddit_results[cur_subreddit] = 1
            else:
                subreddit_results[cur_subreddit] += 1
        elif el["id"] in non_filtered_ids_set:
            non_filtered_data.append(el)
    print(subreddit_results)

    print(f"leftover data has {len(non_filtered_data)} entries")
    with open("./data/hatebert_results/complete_qa_hatebert.json", "w") as outputfile:
        json.dump(non_filtered_data, outputfile)


if __name__ == "__main__":
    hatebert_filtering()
