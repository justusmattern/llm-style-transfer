from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import torch
from transformers import pipeline
from transfer import read_file


def run(file, label):
    texts = read_file
    sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

    predictions = sentiment_analysis(texts)

    rights = 0
    for pred in predictions:
        if pred['label'] == label:
            rights += 1

    print('acc: ', rights/len(predictions))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type = str)
    parser.add_argument('--label', type = int)
    args = parser.parse_args()

    run(file=args.file, label=args.label)