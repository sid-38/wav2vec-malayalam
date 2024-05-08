import functools
import torchaudio
from transformers import pipeline, Wav2Vec2Processor
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import time
import torch
import sys
import asyncio

is_transcribe_over = False
total_chunks = None
current_chunk = None

class MyPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(self, *args, **kwargs):
        self.update_chunk_counter = kwargs.pop('update_chunk_counter', None)
        super().__init__(*args, **kwargs)


    def _forward(self, *args, **kwargs):
        stride = args[0]['stride']
        self.update_chunk_counter(stride[0] - stride[1] - stride[2])
        return super()._forward(*args, **kwargs)

class Transcriber:

    def __init__(self, inp_file, out_file):
        self.inp_file = inp_file
        self.out_file = out_file

    async def transcribe(self):

        start = time.time()

        input_array = None

        with open(self.inp_file, 'rb') as f:
            input_bytes = f.read()
            input_array = ffmpeg_read(input_bytes, 16000)
        print(len(input_array))

        self.total_chunks = len(input_array)
        self.current_chunk = 0

        def update_func(new_val):
            self.current_chunk += new_val

        custom_pipeline = functools.partial(MyPipeline, update_chunk_counter=update_func)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        processor = Wav2Vec2Processor.from_pretrained("gvs/wav2vec2-large-xlsr-malayalam")
        asr = pipeline(
                 model="gvs/wav2vec2-large-xlsr-malayalam", pipeline_class=custom_pipeline
                )

        transcription = asr(input_array, chunk_length_s=30, max_new_tokens=444)
        end = time.time()

        with open(self.out_file, 'w') as f:
            f.write(transcription['text'])
        print(transcription)
        print(f"Took {end-start} seconds")
    
    def get_status(self):
        return (self.current_chunk, self.total_chunks)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('Usage error')
        sys.exit(1)

    inp_file = sys.argv[1]
    out_file = sys.argv[2]

    transcriber = Transcriber(inp_file, out_file)
    asyncio.run(transcriber.transcribe())
