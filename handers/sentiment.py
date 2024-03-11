import torch
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from transformers import RobertaForSequenceClassification, AutoTokenizer

SENTIMENT_DATA = ['Positive', 'Negative', 'Neutral']


class Sentiment:
    def __init__(self):
        self.result_analyzed = []
        self.model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
        self.tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

    def sentiment_analyze(self, transcriptions):
        for sentence in transcriptions:
            input_ids = torch.tensor([self.tokenizer.encode(sentence)])

            with torch.no_grad():
                output_converted = self.model(input_ids)
                self.result_analyzed.append(output_converted.logits.softmax(dim=1).tolist()[0])


def classification_sentence(sentiments_scores):
    n_topics = 3
    model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_assignments = model.fit_transform(sentiments_scores)
    data_array = np.array(topic_assignments)
    topic_assignments = np.sum(data_array, axis=0)
    max_sum_index = np.argmax(topic_assignments)
    return SENTIMENT_DATA[max_sum_index]
