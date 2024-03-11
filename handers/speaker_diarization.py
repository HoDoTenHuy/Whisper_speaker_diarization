import datetime
import numpy as np

from handers.speech2text import ASR
from utils.common import time_fn
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from handers.sentiment import Sentiment, classification_sentence
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


class SpeakerDiarization(ASR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_speaker = kwargs.get('num_speaker')

        self.result_data = None
        self.embeddings = None
        self.best_num_speaker = None
        self.embedding_model = \
            PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=kwargs.get('device'))

    @staticmethod
    def convert_time(secs):
        return datetime.timedelta(seconds=round(secs))

    def segment_embedding(self, segment):
        audio = Audio()
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = segment["end"]
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(self.audio, clip)
        return self.embedding_model(waveform[None])

    def diarization(self):
        self.convert_to_text()
        self.embeddings = np.zeros(shape=(len(self.segments), 192))
        for i, segment in enumerate(self.segments):
            self.embeddings[i] = self.segment_embedding(segment)
        self.embeddings = np.nan_to_num(self.embeddings)
        print(f'Embedding shape: {self.embeddings.shape}')

        if self.num_speaker == 0:
            # Find the best number of speakers
            score_num_speakers = {}

            for self.num_speaker in range(2, 10 + 1):
                clustering = AgglomerativeClustering(self.num_speaker).fit(self.embeddings)
                score = silhouette_score(self.embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[self.num_speaker] = score
            self.best_num_speaker = max(score_num_speakers, key=lambda x: score_num_speakers[x])
            print(f"The best number of speakers: {self.best_num_speaker}"
                  f" with {score_num_speakers[self.best_num_speaker]} score")
        else:
            self.best_num_speaker = self.num_speaker

    def assign_label(self):
        clustering = AgglomerativeClustering(self.best_num_speaker).fit(self.embeddings)
        labels = clustering.labels_
        for i in range(len(self.segments)):
            self.segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    def make_output(self):
        objects = {
            'Start': [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(self.segments):
            if i == 0 or self.segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(self.convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(self.convert_time(self.segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '

        last_segment_index = len(self.segments) - 1
        objects['End'].append(str(self.convert_time(self.segments[last_segment_index - 1]["end"])))
        objects['Text'].append(text)
        self.result_data = objects

    @time_fn
    def run_diarization(self):
        self.diarization()
        self.assign_label()
        self.make_output()


if __name__ == '__main__':
    whisper_model = "base"  # tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2
    number_speakers = 2
    audio_file = f"D:/T-Agent/Whisper_speaker_diarization/audio/audio_3_1.wav"
    device = "cpu"
    input_arg = {'model_whisper': whisper_model, 'audio': audio_file, 'device': device, 'num_speaker': number_speakers,
                 'compute_type': "int8"}
    speaker_diarization = SpeakerDiarization(**input_arg)

    speaker_diarization.run_diarization()
    result_data = speaker_diarization.result_data
    transcriptions = result_data.get('Text')
    sentiment = Sentiment()
    # sentiment.sentiment_analyze(transcriptions)
    # result_data['sentiment'] = sentiment.result_analyzed

    text_length = len(list(result_data.values())[0])
    for i in range(text_length):
        for key in result_data:
            print(result_data[key][i], end=" ")
        print('\n')
    # print(result_data)

    speaker_dict = {}

    # Iterate through the data
    for speaker, text in zip(result_data["Speaker"], result_data["Text"]):
        if speaker not in speaker_dict:
            speaker_dict[speaker] = []
        speaker_dict[speaker].append(text)

    for key, value in speaker_dict.items():
        sentiment.sentiment_analyze(value)
        sentiment_result = classification_sentence(sentiment.result_analyzed)
        print(key, sentiment_result)
