import time
import torch
import logging

logging.basicConfig(level=logging.INFO)

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)

from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig, \
    CodeGenTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory, ConversationBufferMemory, \
    ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate

class LLMEngine:
    def __init__(self):
        self.llm_chain = None
        self.eos = False
        self.infer_time = 0
        self.conversation = None


    def init_model(self):
        #model = "microsoft/phi-2"
        model_name = 'openbmb/MiniCPM-2B-sft-fp32'
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True,
                                                  use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True,
                                                     device_map={"": 0})
        
        pipe = pipeline(
            "text-generation",
            model=model,
            do_sample=True,
            tokenizer=tokenizer,
            max_length=256,
            max_new_tokens=50,
            temperature=0.3,
            top_p=0.3,
            repetition_penalty=1.2
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)

        template = """The following is a friendly conversation between a human and an AI. The AI provides specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. Make the answer brief and valueable.
examples: 

User: How are you?
AI: I can't complain but sometimes I still do.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""

        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        self.conversation = ConversationChain(
            llm=local_llm, verbose=True, prompt=prompt, memory=ConversationBufferWindowMemory(k=3)
        )

        logging.info("[LLM INFO:] Loaded LLM Engine.")

    def run(
            self,
            transcription_queue=None,
            llm_queue=None,
            audio_queue=None,
            max_output_len=50,
            max_attention_window_size=4096,
            num_beams=1,
            streaming=False,
            streaming_interval=4,
            debug=True,
    ):
        logging.info('llm service started')

        conversation_history = {}
        self.init_model()


        while True:
            # Get the last transcription output from the queue
            transcription_output = transcription_queue.get()
            if transcription_queue.qsize() != 0:
                continue

            if transcription_output["uid"] not in conversation_history:
                conversation_history[transcription_output["uid"]] = []

            prompt = transcription_output['prompt'].strip()

            # if prompt is same but EOS is True, we need that to send outputs to websockets

            if transcription_output["eos"]:
                start = time.time()
                llm_output = self.conversation.predict(input=prompt)
                llm_output = llm_output.replace('\n', "<br>")
                self.infer_time = time.time() - start
                self.eos = transcription_output["eos"]
                logging.info(f"[LLM INFO:] Running LLM Inference  with "
                             f"WhisperLive prompt: {prompt}, eos: {self.eos},"
                             f" llm output: {llm_output}, infer_time: {self.infer_time}")
                llm_queue.put({
                    "uid": transcription_output["uid"],
                    "llm_output": llm_output,
                    "eos": self.eos,
                    "latency": self.infer_time
                })
                audio_queue.put({"llm_output": llm_output, "eos": self.eos})
 
 
def main():
    llm = LLMEngine()
    llm.init_model()

    print(llm.conversation_with_summary.predict(input='hello, how are you today'))

if "__main__" == __name__:
    main()

