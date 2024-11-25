import json
from tqdm import tqdm
from transformers import pipeline


def hate_classification() -> None:
    hate_classifier = pipeline(
        "text-classification", model="badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification", device=0)
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
    with open("./data/hatebert_results/complete_qa_hatebert.json", "r") as input_file:
        data = json.load(input_file)

    # We only do hate classification on the preferred answer in the dataset
    texts = [el["answer_1"] for el in data]
    questions = [el["text"] + " " + el["context"] for el in data]
    predictions = []
    for idx, el in tqdm(enumerate(texts), total=len(texts)):
        response = hate_classifier(el, **tokenizer_kwargs)
        prediction = response[0]["label"]
        if prediction == "HATE-SPEECH" or prediction == "OFFENSIVE-LANGUAGE":
            new_result = {
                "idx": idx,
                "questions": questions[idx],
                "prediction": el,
                "reason": prediction
            }
            predictions.append(new_result)
    with open(f"./data/hate_classification_results/hate_classifier_results.json", "w") as output_file:
        json.dump(predictions, output_file)


if __name__ == "__main__":
    hate_classification()
