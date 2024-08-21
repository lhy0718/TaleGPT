import gc

import gradio as gr
import torch
from gradio import ChatMessage
from transformers import AutoTokenizer, TextIteratorStreamer, pipeline

MODEL = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
streamer = TextIteratorStreamer(
    tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
)

pipe = pipeline(
    "text-generation",
    model=MODEL,
    tokenizer=tokenizer,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_new_tokens=512,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    streamer=streamer,
    eos_token_id=tokenizer.eos_token_id,
)

NO_INPUT_WARNING = "아무것도 입력하지 않았습니다. 동화 제목을 입력해주세요."
CONTINUE_PROMPT = "이전 동화를 이어서 출력합니다."


def answer(user_input, history, top_p, top_k, temperature):
    if user_input == "":
        if history[-1][1] == NO_INPUT_WARNING:
            yield NO_INPUT_WARNING
        else:
            yield CONTINUE_PROMPT
    else:
        gc.collect()
        torch.cuda.empty_cache()
    history.append(ChatMessage(role="user", content=user_input))
    print(history)

    generate_kwargs = dict(
            history,
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
    stop_btn = gr.Button("멈춤", render=False, variant="stop")

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
        concurrency_limit=2,
    )

demo.queue().launch(share=True, auth=("automl", "208217"))
