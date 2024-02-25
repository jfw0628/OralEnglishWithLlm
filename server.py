import torch

import multiprocessing

from whisper_service.whisper_service import TranscriptionServer
from llm_service.llm_service import  LLMEngine
from tts_service.tts_serice import SpeechTTS



from multiprocessing import Process, Manager, Value, Queue


def main():
    transcription_queue = Queue()
    llm_queue = Queue()
    audio_queue = Queue()



    whisper_server = TranscriptionServer()
    whisper_process = multiprocessing.Process(
        target=whisper_server.run,
        args=(
            "0.0.0.0",
            6006,
            transcription_queue,
            llm_queue,
        )
    )
    whisper_process.start()

    tts_provider = SpeechTTS()
    tts_process = multiprocessing.Process(
        target=tts_provider.run,
        args=(
            '0.0.0.0', 
            8765,
            audio_queue
        )
    )
    tts_process.start()

    llm_provider = LLMEngine()
    llm_process = multiprocessing.Process(
        target=llm_provider.run,
        args=(
            transcription_queue,
            llm_queue,
            audio_queue,
        )
    )
    llm_process.start()


    whisper_process.join()
    tts_process.join()
    llm_process.join()



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()