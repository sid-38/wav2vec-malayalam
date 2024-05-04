import torchaudio
from transformers import pipeline, Wav2Vec2Processor
import time
import torch
import sys

if len(sys.argv) < 3:
    print('Usage error')
    sys.exit(1)

inp_file = sys.argv[1]
out_file = sys.argv[2]

resampler = torchaudio.transforms.Resample(48_000, 16_000)

def speech_file_to_array_fn(file):
  speech_array, sampling_rate = torchaudio.load(file)
  return resampler(speech_array).squeeze().numpy()

start = time.time()

input_batch = speech_file_to_array_fn(inp_file)
print(input_batch)
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained("gvs/wav2vec2-large-xlsr-malayalam")
asr = pipeline(
         model="gvs/wav2vec2-large-xlsr-malayalam",
        )

transcription = asr(inp_file, chunk_length_s=30, max_new_tokens=444)
end = time.time()

with open(out_file, 'w') as f:
    f.write(transcription['text'])
print(transcription)
print(f"Took {end-start} seconds")

