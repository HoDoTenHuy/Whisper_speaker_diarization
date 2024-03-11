import logging

from fastapi.routing import APIRouter
from utils.config_yaml import ConfigManager
from handers.sentiment import Sentiment, classification_sentence
from handers.speaker_diarization import SpeakerDiarization
from utils.uitls import Whisper_Model, Device
from fastapi import FastAPI, File, UploadFile
from utils.common import save_upload_file_tmp, delete_tmp_file

config_manager = ConfigManager()
config = config_manager.get_config()
model_config = config.get('object_config')

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post('/speech_to_text')
def convert_speech2text(whisper_model: Whisper_Model, audio_file: UploadFile, device: Device, number_speakers: int = 2):
    audio_file = save_upload_file_tmp(audio_file)
    input_arg = {'model_whisper': whisper_model, 'audio': str(audio_file), 'device': device,
                 'num_speaker': number_speakers, 'compute_type': "int8"}

    speaker_diarization = SpeakerDiarization(**input_arg)
    speaker_diarization.convert_to_text()
    result_data = speaker_diarization.segments
    delete_tmp_file(str(audio_file))
    return result_data


@router.post('/speaker-identification')
def speaker_identification(whisper_model: Whisper_Model, audio_file: UploadFile, device: Device,
                           number_speakers: int = 2):
    audio_file = save_upload_file_tmp(audio_file)
    input_arg = {'model_whisper': whisper_model, 'audio': str(audio_file), 'device': device,
                 'num_speaker': number_speakers, 'compute_type': "int8"}

    speaker_diarization = SpeakerDiarization(**input_arg)
    speaker_diarization.run_diarization()
    result_data = speaker_diarization.result_data

    text_length = len(list(result_data.values())[0])
    data_text = []
    for i in range(text_length):
        temp_text = []
        for key in result_data:
            print(result_data[key][i], end=" ")
            temp_text.append(result_data[key][i])
        data_text.append(temp_text)
    delete_tmp_file(str(audio_file))
    return data_text


@router.post('/sentiment_analyze')
def sentiment_analyze(whisper_model: Whisper_Model, audio_file: UploadFile, device: Device,
                      number_speakers: int = 2):
    audio_file = save_upload_file_tmp(audio_file)
    input_arg = {'model_whisper': whisper_model, 'audio': str(audio_file), 'device': device,
                 'num_speaker': number_speakers, 'compute_type': "int8"}

    sentiment = Sentiment()
    speaker_diarization = SpeakerDiarization(**input_arg)
    speaker_diarization.run_diarization()
    result_data = speaker_diarization.result_data

    speaker_dict = {}
    for speaker, text in zip(result_data["Speaker"], result_data["Text"]):
        if speaker not in speaker_dict:
            speaker_dict[speaker] = []
        speaker_dict[speaker].append(text)

    sentiment_data = {}
    for key, value in speaker_dict.items():
        sentiment.sentiment_analyze(value)
        sentiment_result = classification_sentence(sentiment.result_analyzed)
        sentiment_data.update({f"{key}": sentiment_result})
    delete_tmp_file(str(audio_file))
    return sentiment_data
