from utils.common import time_fn
from faster_whisper import WhisperModel


class ASR:
    def __init__(self, *args, **kwargs):
        self.model_whisper = kwargs.get('model_whisper')
        self.audio = kwargs.get('audio')
        self.device = kwargs.get('device')
        self.compute_type = kwargs.get('compute_type')

        self.segments = []
        self.language = None
        self.language_probability = None

    @time_fn
    def convert_to_text(self):
        # Run on GPU with FP16
        model = WhisperModel(self.model_whisper, device=self.device, compute_type=self.compute_type)

        segments_raw, info = model.transcribe(self.audio, beam_size=5)

        self.language, self.language_probability = info.language, info.language_probability
        print("Detected language '%s' with probability %f" % (self.language, self.language_probability))

        # Convert back to original openai format
        i = 0
        for segment_chunk in segments_raw:
            chunk = {"start": segment_chunk.start, "end": segment_chunk.end, "text": segment_chunk.text}
            self.segments.append(chunk)
            i += 1
