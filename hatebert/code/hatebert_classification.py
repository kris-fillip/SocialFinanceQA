import re
import emoji
import json
import os
import torch
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, BertConfig
import numpy as np
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer


def preprocess_regex(text: str) -> str:
    text = re.sub(r'#([^ ]*)', r'\1', text)
    text = re.sub(r'https.*[^ ]', 'URL', text)
    text = re.sub(r'http.*[^ ]', 'URL', text)
    text = re.sub(r'@([^ ]*)', '@USER', text)
    text = emoji.demojize(text)
    text = re.sub(r'(:.*?:)', r' \1 ', text)
    text = re.sub(' +', ' ', text)
    return text


def preprocess_reddit_submissions(tokenizer: BertTokenizer, input_dir: str) -> tuple[list, list, list, list]:
    with open(input_dir, "r") as input_file:
        data = json.load(input_file)

    texts = []
    for el in tqdm(data, total=len(data)):
        texts.append(preprocess_regex(el["answer_1"]))
    text_ids = [str(idx) for idx, el in enumerate(texts)]
    # List of all tokenized tweets
    test_input_ids = []
    # For every tweet in the test set
    for sent in tqdm(texts, total=len(texts)):
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=500,

            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded tweet to the list.
        test_input_ids.append(encoded_sent)

    # # Pad our input tokens with value 0.
    # # "post" indicates that we want to pad and truncate at the end of the sequence,
    # # as opposed to the beginning.
    test_input_ids = pad_sequences(test_input_ids, maxlen=500, dtype="long",
                                   value=tokenizer.pad_token_id, truncating="pre", padding="pre")

    # Create attention masks
    # The attention mask simply makes it explicit which tokens are actual words versus which are padding
    test_attention_masks = []

    # For each tweet in the test set
    for sent in tqdm(test_input_ids, total=len(test_input_ids)):
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        test_attention_masks.append(att_mask)

    # Return the list of encoded tweets, the list of labels and the list of attention masks
    return test_input_ids, test_attention_masks, text_ids, texts


def hatebert_classification():
    # ======================================================================================================================
    # Part of the code comes from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    # ======================================================================================================================
    # ---------------------------- Main ----------------------------

    # Directory where the pretrained model can be found
    model_dir = "./pipeline/hatebert/HateBERT_offenseval"
    input_dir = "./data/question_answers/no_gilding_and_awards_percentile90/complete_qa.json"
    output_dir = "./data/hatebert_results"

    # -----------------------------
    # Load Pre-trained BERT model
    # -----------------------------
    config_class, model_class, tokenizer_class = (
        BertConfig, BertForSequenceClassification, BertTokenizer)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load a trained model and vocabulary pre-trained for specific language
    print("Loading model from: '" + model_dir + "', it may take a while...")

    # Load pre-trained Tokenizer from directory, change this to load a tokenizer from ber package
    tokenizer = tokenizer_class.from_pretrained(model_dir)

    # Load Bert for classification 'container'
    model = BertForSequenceClassification.from_pretrained(
        model_dir,  # Use pre-trained model from its directory, change this to use a pre-trained model from bert
        # The number of output labels--2 for binary classification.
        num_labels=2,
        # You can increase this for multi-class tasks.
        # Whether the model returns attentions weights.
        output_attentions=False,
        # Whether the model returns all hidden-states.
        output_hidden_states=False,
    )

    # Set the model to work on CPU if no GPU is present
    # model.to(device)
    print("Bert for classification model has been loaded!")

    # --------------------------------------------------------------------
    # -------------------------- Load test data --------------------------
    # --------------------------------------------------------------------

    # The loading eval data return:
    # - input_ids:         the list of all tweets already tokenized and ready for bert (with [CLS] and [SEP)
    # - labels:            the list of labels, the i-th index corresponds to the i-th position in input_ids
    # - attention_masks:   a list of [0,1] for every input_id that represent which token is a padding token and which is not
    # input_ids, labels, attention_masks = load_eval_data(language, tokenizer)

    # Load Offenseval 2018, Train,Test already divided into Train/Test set

    prediction_inputs, prediction_masks, tweet_ids, texts = preprocess_reddit_submissions(
        tokenizer, input_dir)

    print("preprocessed submissions")

    # Tweets
    prediction_inputs = torch.tensor(prediction_inputs)

    # Attention masks
    prediction_masks = torch.tensor(prediction_masks)

    label_encoder = preprocessing.LabelEncoder()
    targets = label_encoder.fit_transform(tweet_ids)
    # targets: array([0, 1, 2, 3])

    prediction_ids = torch.as_tensor(targets)
    # targets: tensor([0, 1, 2, 3])

    # Set the batch size.
    batch_size = 32

    # Create the DataLoader.
    prediction_data = TensorDataset(
        prediction_inputs, prediction_masks, prediction_ids)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    print('Predicting labels for {:,} test sentences...'.format(
        len(prediction_inputs)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions = []

    # Predict
    for batch in tqdm(prediction_dataloader):
        # Add batch to GPU
        # batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_tweet_ids = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        flat_logits = np.argmax(logits, axis=1).flatten()
        # Get tweet ids for prediction output
        ids = label_encoder.inverse_transform(b_tweet_ids.cpu().numpy())
        # Store predictions and true labels
        predictions.extend(list(zip(ids, flat_logits)))
        # print(predictions)

    print("Saving predictions...")
    # Print the list of prediction & store for evaluation
    result_dict = {}
    filtered_answers = []
    keep_answers = []
    for idx, i in enumerate(predictions):
        if str(i[1]) == '1':
            filtered_answers.append(texts[idx])
        else:
            keep_answers.append(texts[idx])
        result_dict.setdefault(i[0], str(i[1]).replace(
            '0', 'NOT').replace('1', 'OFF'))

    print(
        f"There were {len(filtered_answers)} answers filtered based on offensives speech")
    print(f"{len(keep_answers)} answers left in dataset")
    with open(os.path.join(output_dir, "offense.json"), "w") as outfile:
        json.dump(result_dict, outfile)
    with open(os.path.join(output_dir, "offense_filtered.json"), "w") as outfile:
        json.dump(filtered_answers, outfile)
    with open(os.path.join(output_dir, "offense_keep.json"), "w") as outfile:
        json.dump(keep_answers, outfile)


if __name__ == "__main__":
    hatebert_classification()
