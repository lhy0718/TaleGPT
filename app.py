import gc
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

NO_INPUT_WARNING = "아무것도 입력하지 않았습니다. 동화 제목을 입력해주세요."
INITIAL_PROMPT = "입력되는 제목을 바탕으로 대한 아동을 타겟으로 하는 동화 또는 각종 판타지 요소가 난무하는 판타지 소설을 출력합니다."
CONTINUE_PROMPT = "출력이 끝난 부분부터 동화를 이어서 출력합니다."


def convert_history_to_messages(history: list = []) -> list:
    messages = [{"role": "system", "content": INITIAL_PROMPT}]
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


def answer(user_input, history, top_p, top_k, temperature):
    if (len(history) == 0 or history[-1][1] == NO_INPUT_WARNING) and user_input == "":
        yield NO_INPUT_WARNING
    else:
        gc.collect()
        torch.cuda.empty_cache()

        if user_input.strip() == "":  # 이전 동화를 이어서 출력
            history = convert_history_to_messages(history[-4:])
            history.append({"role": "system", "content": CONTINUE_PROMPT})
        else:  # 새로운 동화 생성
            history = convert_history_to_messages()
            history.append({"role": "user", "content": user_input})

        input_ids = tokenizer.apply_chat_template(
            history, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        streamer = TextIteratorStreamer(
            tokenizer, timeout=3, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = dict(
            inputs=input_ids,
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        partial_message = ""
        for new_text in streamer:
            partial_message += new_text
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

demo.queue(default_concurrency_limit=2).launch(
    # share=True,
    # auth=("automl", "208217"),
)
