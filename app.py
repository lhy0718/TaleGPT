import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from threading import Thread

import gradio as gr
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList,
                          TextIteratorStreamer)

MODEL = "beomi/KoAlpaca-Polyglot-12.8B"
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    load_in_8bit=True,
    revision="8bit",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2, 6, 31] # "<|endoftext|>": 2, "#": 6, "<": 31
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    

def convert_history_item_to_message(history_item: list) -> str:
    return f"### 명령어:\n{history_item[0].strip()}\n### 판타지 동화 출력:\n{history_item[1].split('#')[0].strip()}"


def answer(message, history):
    pre_system_message = '''
### 맥락:
사용자의 명령에 따른 동화 판타지 소설을 출력합니다.
'''
    curr_system_message = '''
### 명령어:
다음 명령에 대한 아동을 타겟으로 하는 500자 정도의 동화 또는 검/마법/기사/요정 등이 난무하는 판타지 소설을 출력해줘.
'''

    stop = StopOnTokens()

    history_transformer_format = history + [[message, ""]]
    
    messages = (
        pre_system_message
        +"\n".join(map(convert_history_item_to_message, history_transformer_format[:-1])) 
        + curr_system_message 
        + convert_history_item_to_message(history_transformer_format[-1])
    )
    print("\n========== Input Messages")
    print(messages)

    model_inputs = tokenizer([messages], return_token_type_ids=False, return_tensors="pt")
    if torch.cuda.is_available():
        model_inputs = model_inputs.to('cuda')
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        top_k=1000,
        temperature=0.5,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message



gr.ChatInterface(
    answer,
    examples=[""],
    title="TaleGPT (동화 / 판타지 소설 생성 인공지능)",
    description="© 중앙대학교 기계학습자동화연구실 - CAU AutoML Lab"
).queue().launch(debug=True)