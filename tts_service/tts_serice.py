import functools
import time
import scipy
import logging
import numpy as np
from io import BytesIO

logging.basicConfig(level = logging.INFO)

from websockets.sync.server import serve

from TTS.api import TTS
import torch


# Get device


class SpeechTTS:
    def __init__(self):
        pass
    
    def initialize_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
        self.last_llm_response = None

    def run(self, host, port, audio_queue=None):
        # initialize and warmup model
        print(1, host, port)
        self.initialize_model()
        self.tts.tts("Hello, I am warming up.")

        with serve(
            functools.partial(self.start_tts, audio_queue=audio_queue), 
            host, port
            ) as server:
            server.serve_forever()

    def start_tts(self, websocket, audio_queue=None):
        self.eos = False
        self.output_audio = None

        while True:
            llm_response = audio_queue.get()
            if audio_queue.qsize() != 0:
                continue

            # llm_response = {'llm_output': "hello, how are you?", 'eos': True}
            # check if this websocket exists
            try:
                websocket.ping()
            except Exception as e:
                del websocket
                audio_queue.put(llm_response)
                break
            
            llm_output = llm_response["llm_output"]
            self.eos = llm_response["eos"]

            try:
                start = time.time()
                wav = self.tts.tts(llm_output)
                wav_norm = np.array(wav) * (32767 / max(0.01, np.max(np.abs(wav))))
                wav_norm = wav_norm.astype(np.int16)
                wav_buffer = BytesIO()
                scipy.io.wavfile.write(wav_buffer, 24000, wav_norm)
                wav_buffer.seek(0)
                self.output_audio = wav_buffer.read() 
                # self.output_audio = np.array(wav)
                inference_time = time.time() - start
                logging.info(f"[WhisperSpeech INFO:] TTS inference done in {inference_time} ms.\n")
            except TimeoutError:   
                pass
            
            if self.eos and self.output_audio is not None:
                try:
                    #websocket.send(self.output_audio.tobytes())
                    websocket.send(self.output_audio)

                except Exception as e:
                    logging.error(f"[WhisperSpeech ERROR:] Audio error: {e}")

# wst = WhisperSpeechTTS()
# wst.run('0.0.0.0', 8765)