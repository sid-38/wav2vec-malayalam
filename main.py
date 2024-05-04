import os
import torchaudio
from transformers import pipeline, WhisperProcessor, Pipeline, Wav2Vec2Processor
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import time
from pydub import AudioSegment
from pydub.utils import make_chunks
import json
import numpy as np
import torch
from tqdm import tqdm

# class LogWrapper:
#     def __init__(self, obj):
#         print("Class of the object wrapped in logger is ", obj.__class__)
#         self._my_weird_obj = obj

#     def __getattr__(self, item):
#         print("Calling", item)
#         return self._my_weird_obj.__getattribute__(item)

# class Foo:
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b

#     def sum(self):
#         return self.a + self.b
    
#     def sum_custom(self, a, b):
#         return a + b

# foo = Foo(2,3)
# foo_with_log = LogWrapper(foo)
# test = foo_with_log.a
# sum_val = foo_with_log.sum()
# print(type(foo_with_log))
# class MyPipeline(AutomaticSpeechRecognitionPipeline):
#     def __init__(self, *args, **kwargs):
#         self.pbar = tqdm()
#         super().__init__(*args, **kwargs)

#     def _sanitize_parameters(self, *args, **kwargs):
#         return super()._sanitize_parameters(*args, **kwargs)

#     def preprocess(self, *args, **kwargs):
#         self.pbar.total = len(args[0])
#         self.pbar.refresh()
#         return super().preprocess(*args, **kwargs)

#     def _forward(self, *args, **kwargs):
#         stride = args[0]['stride']
#         print(stride)
#         self.pbar.update(stride[1])
#         return super()._forward(*args, **kwargs)

#     def postprocess(self, *args, **kwargs):
#         return super().postprocess(*args, **kwargs)

#     def __del__(self, *args, **kwargs):
#         self.pbar.close()
#         super().__del__(*args, **kwargs)
resampler = torchaudio.transforms.Resample(48_000, 16_000)

def speech_file_to_array_fn(file):
  speech_array, sampling_rate = torchaudio.load(file)
  return resampler(speech_array).squeeze().numpy()

start = time.time()

# input_array = None
# with open("audio_combined.mp3", 'rb') as f:
#     input_bytes = f.read()
#     input_array = ffmpeg_read(input_bytes, 16000)
input_batch = speech_file_to_array_fn("./audio_combined.mp3")
print(input_batch)
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# processor = WhisperProcessor.from_pretrained("thennal/whisper-medium-ml")
# processor = WhisperProcessor.from_pretrained("kavyamanohar/whisper-small-malayalam")
# processor = WhisperProcessor.from_pretrained("gvs/wav2vec2-large-xlsr-malayalam")
processor = Wav2Vec2Processor.from_pretrained("gvs/wav2vec2-large-xlsr-malayalam")
# forced_decoder_ids = processor.get_decoder_prompt_ids(language="ml", task="transcribe")
asr = pipeline(
        # "automatic-speech-recognition", model="thennal/whisper-medium-ml",
        # "automatic-speech-recognition", model="kavyamanohar/whisper-small-malayalam",
         model="gvs/wav2vec2-large-xlsr-malayalam",
        )

# audio_whole = AudioSegment.from_file("./audio1.wav")
# audio_array = np.array(audio_whole.get_array_of_samples())
# audio_array_with_log = LogWrapper(audio_array)
# chunk_length_ms = 25000
# chunks = make_chunks(audio_whole, chunk_length_ms)

# with open("./output.txt", 'a') as f:
#     chunk_len = len(chunks)
#     for i, chunk in enumerate(chunks):
#         transcription = asr(np.array(chunk.get_array_of_samples()), chunk_length_s=30, max_new_tokens=444, return_timestamps=False,  generate_kwargs={
#             "forced_decoder_ids": forced_decoder_ids, 
#             "do_sample": True,
#             })
#         print(i, "/", chunk_len, "->",  transcription['text'])
#         f.write(transcription['text'])
# audio = "./audio_combined.mp3"

# transcription = asr("./audio1.wav", chunk_length_s=10, max_new_tokens=444, return_timestamps=False,  generate_kwargs={
#             "forced_decoder_ids": forced_decoder_ids, 
#             "do_sample": True,
#             })
transcription = asr("./audio_combined.mp3", chunk_length_s=30, max_new_tokens=444)
end = time.time()
with open("output1.txt", 'w') as f:
    f.write(transcription['text'])
print(transcription)
print(f"Took {end-start} seconds")
