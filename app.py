import os
from tkinter import NO

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gc
import json
from threading import Thread

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

INITIAL_PROMPT = "### 명령어:\n다음 명령에 대한 아동을 타겟으로 하는 동화 또는 각종 판타지 요소가 난무하는 판타지 소설을 출력합니다.\n"
GEN_PROMPT_FORMAT = "### 제목:\n{title}\n### 판타지 동화 출력:\n{story}"
CONTINUEW_GEN_PROMPT_FORMAT = "### 이전과 이어서 동화 계속 출력:\n{story}"
NO_INPUT_WARNING = "아무것도 입력하지 않았습니다. 동화 제목을 입력해주세요."

MODEL = "beomi/KoAlpaca-Polyglot-12.8B"  # 제일 좋음
# MODEL = "beomi/Solar-Ko-Recovery-11B"
# MODEL = "beomi/OPEN-SOLAR-KO-10.7B"
# MODEL = "beomi/Llama-3-Open-Ko-8B"
# MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

if torch.cuda.is_available():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        quantization_config=quantization_config,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL)

tokenizer = AutoTokenizer.from_pretrained(MODEL)


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [tokenizer.eos_token_id] + tokenizer.convert_tokens_to_ids(
            ["#", "<"]
        )
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def convert_history_item_to_message(history_item: list) -> str:
    title, story = history_item
    if story == NO_INPUT_WARNING:
        return ""
    title = title or ""
    story = story or ""
    if title == "":
        return CONTINUEW_GEN_PROMPT_FORMAT.format(story=story)
    return GEN_PROMPT_FORMAT.format(title=title, story=story)


def answer(user_input, history, top_p, top_k, temperature):
    if (len(history) == 0 or history[-1][1] == NO_INPUT_WARNING) and user_input == "":
        yield NO_INPUT_WARNING
    else:
        gc.collect()
        torch.cuda.empty_cache()

        with open("fewshot.json") as f:
            fewshot = json.load(f)
            history = fewshot + history

        stop = StopOnTokens()

        history += [[user_input, ""]]  # history: [[title, story], ...]

        input_text = INITIAL_PROMPT + "\n".join(
            [
                convert_history_item_to_message(item) for item in history[-6:]
            ]  # 최근 5개의 fewshot + user_input
        )

        print("\n========== Input Messages")
        print(input_text)

        model_inputs = tokenizer(
            [input_text], return_token_type_ids=False, return_tensors="pt"
        )
        if torch.cuda.is_available():
            model_inputs = model_inputs.to("cuda")

        streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )

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
            stopping_criteria=StoppingCriteriaList([stop]),
        )

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        partial_message = ""
        for new_token in streamer:
            if new_token not in ["<", "#"]:
                partial_message += new_token
                yield partial_message


# Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    chatbot = gr.Chatbot(
        show_copy_button=True,
        avatar_images=("images/user.png", "images/bot.png"),
        render=False,
    )
    textbox = gr.Textbox(
        placeholder="여기에 입력하세요. (아무것도 입력하지 않으면 이전 동화를 이어서 출력합니다.)",
        scale=7,
        render=False,
    )
    top_p_slider = gr.Slider(0, 1, value=0.95, label="단어 선택의 다양성", render=False)
    top_k_slider = gr.Slider(
        0, 5000, value=2000, label="단어 선택의 가짓수", render=False
    )
    temperature_slider = gr.Slider(0, 1, value=1, label="창의성", render=False)

    gr.ChatInterface(
        answer,
        chatbot=chatbot,
        textbox=textbox,
        examples=[
            ["금화를 지키는 용"],
            ["마법 동물원의 비밀"],
            ["전설의 검"],
        ],
        title="TaleGPT (동화 / 판타지 소설 생성 인공지능)",
        description="© 중앙대학교 기계학습자동화연구실 - CAU AutoML Lab",
        submit_btn="제출",
        stop_btn="멈춤",
        retry_btn="🔄 다시 시도",
        undo_btn="↩️ 되돌리기",
        clear_btn="🗑️ 지우기",
        additional_inputs=[top_p_slider, top_k_slider, temperature_slider],
    )

demo.queue().launch(share=False, auth=("automl", "208217"), server_name="0.0.0.0")
