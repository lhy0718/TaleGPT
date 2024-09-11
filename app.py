import gradio as gr
from huggingface_hub import InferenceClient

NO_INPUT_WARNING = "아무것도 입력하지 않았습니다. 동화 제목을 입력해주세요."
INITIAL_PROMPT = "입력되는 제목을 바탕으로 대한 아동을 타겟으로 하는 동화 또는 각종 판타지 요소가 난무하는 판타지 소설을 출력합니다."
CONTINUE_PROMPT = "출력이 끝난 부분부터 동화를 이어서 출력합니다."

client = InferenceClient(base_url="http://127.0.0.1:8080")


def convert_history_to_messages(history: list = []) -> list:
    messages = [{"role": "system", "content": INITIAL_PROMPT}]
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


def inference(user_input, history, top_p, temperature):
    if (len(history) == 0 or history[-1][1] == NO_INPUT_WARNING) and user_input == "":
        yield NO_INPUT_WARNING
    else:
        if user_input.strip() == "":  # 이전 동화를 이어서 출력
            history = convert_history_to_messages(
                history[-4:]
            )  # 마지막 4개의 대화만 이어서 출력
            history.append({"role": "system", "content": CONTINUE_PROMPT})
        else:  # 새로운 동화 생성
            history = convert_history_to_messages()
            history.append({"role": "user", "content": user_input})

        partial_message = ""
        output = client.chat.completions.create(
            messages=history,
            stream=True,
            max_tokens=1024,
            temperature=temperature,
            top_p=top_p,
        )

        for chunk in output:
            partial_message += chunk.choices[0].delta.content
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
        inference,
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
        additional_inputs=[top_p_slider, temperature_slider],
    )

demo.queue(default_concurrency_limit=2).launch(
    share=True,
    # auth=("automl", "208217"),
)
