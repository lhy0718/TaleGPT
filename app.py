import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from threading import Thread

import gradio as gr
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList,
                          TextIteratorStreamer, BitsAndBytesConfig)

MODEL = "beomi/KoAlpaca-Polyglot-12.8B"
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        load_in_8bit=True,
        revision="8bit",
        # torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    
tokenizer = AutoTokenizer.from_pretrained(MODEL)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2, 6, 31] # "<|endoftext|>": 2, "#": 6, "<": 31
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    

def convert_history_item_to_message(history_item: list) -> str:
    # return f"### 명령어:\n{history_item[0].strip()}\n### 출력:\n{history_item[1].split('#')[0].strip()}"
    return f"### 제목: {history_item[0].strip()}\n### 출력: {history_item[1].split('#')[0].strip()}"


def answer(message, history, top_p, top_k, temperature):
    with open('prompt.txt') as f:
        prompt = f.read()
    pre_system_message = prompt
    curr_system_message = '''
### 명령어:
다음 제목의 동요를 생성하세요.
'''

    stop = StopOnTokens()

    message = message or ""
    history_transformer_format = history + [[message, ""]]
    
    messages = (
        pre_system_message
        + "\n".join(map(convert_history_item_to_message, history_transformer_format[:-1])) 
        + curr_system_message 
        + convert_history_item_to_message(history_transformer_format[-1])
    )
    print("\n========== Input Messages")
    print(messages)

    model_inputs = tokenizer([messages], return_token_type_ids=False, return_tensors="pt")
    if torch.cuda.is_available():
        model_inputs = model_inputs.to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    for new_token in streamer:
        if new_token not in ["<", "#"]:
            partial_message += new_token
            yield partial_message

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    chatbot = gr.Chatbot(
        show_copy_button=True,
        avatar_images=("images/user.png", "images/bot.png"),
        render=False,
    )
    textbox = gr.Textbox(
        placeholder = "여기에 입력하세요.",
        scale=7,
        render=False,
    )
    top_p_slider = gr.Slider(0, 1, value=0.95, label="단어 선택의 다양성", render=False)
    top_k_slider = gr.Slider(0, 5000, value=2000, label="단어 선택의 가짓수", render=False)
    temperature_slider = gr.Slider(0, 1, value=1, label="창의성", render=False)
    
    gr.ChatInterface(
        answer,
        chatbot=chatbot,
        textbox=textbox,
        examples=[
            # ["금화를 지키는 용"],
            # ["마법 동물원의 비밀"],
            # ["전설의 검"],
            ["달님"],
            ["용이 나타났다!"],
            ["가족의 사랑"],
        ],
        # title="TaleGPT (동화 / 판타지 소설 생성 인공지능)",
        title="LyricGPT",
        description="© 중앙대학교 기계학습자동화연구실 - CAU AutoML Lab",
        submit_btn="제출",
        stop_btn="멈춤",
        retry_btn="🔄 다시 시도",
        undo_btn="↩️ 되돌리기",
        clear_btn="🗑️ 지우기",
        additional_inputs=[top_p_slider, top_k_slider, temperature_slider]
    )
    
    # with open('createpopup.js') as f:
    #     script = f.read()
    # gr.HTML(f'<script type="text/javascript">{script}</script>')
    # print_button = gr.Button(
    #     value="🖨️ 프린트",
    #     size="sm",
    # )
    # print_button.click(fn=None, _js="createPopupWithText()") # not working

demo.queue().launch(share=True)