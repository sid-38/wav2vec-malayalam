import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time

start = time.time()
# test_dataset = <load-test-split-of-combined-dataset> # Details on loading this dataset in the evaluation section
file_path = "./audio_combined.mp3"

processor = Wav2Vec2Processor.from_pretrained("gvs/wav2vec2-large-xlsr-malayalam")
model = Wav2Vec2ForCTC.from_pretrained("gvs/wav2vec2-large-xlsr-malayalam")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(file_path):
  speech_array, sampling_rate = torchaudio.load(file_path)
  return resampler(speech_array).squeeze().numpy()

# test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(speech_file_to_array_fn(file_path)[:2], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
  logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

pred =  processor.batch_decode(predicted_ids)
with open("output.txt", 'w') as f:
    for p in pred:
        f.write(p)
end = time.time()
print(f"Took {end-start} seconds")
