"""Gradio chat UI for Makan Where?  Run:  python app.py"""

import gradio as gr

from src.pipeline import MakanSession


def _chatbot() -> gr.Chatbot:
    try:  # Gradio 5 needs type="messages"; Gradio 6 uses messages by default
        return gr.Chatbot(type="messages", height=520, label="Makan Where?")
    except TypeError:
        return gr.Chatbot(height=520, label="Makan Where?")


def respond(message, chat, session):
    if session is None:
        session = MakanSession()
    if not message or not message.strip():
        return "", chat, session, gr.update()
    reply, debug = session.step(message)
    chat = (chat or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", chat, session, debug


def reset(session):
    if session is not None:
        session.reset()
    return [], session, {}


with gr.Blocks(title="Makan Where?") as demo:
    gr.Markdown("# Makan Where? 🍜\nSinglish-aware restaurant recommender for Singapore (stub pipeline).")
    session = gr.State(None)
    with gr.Row():
        with gr.Column(scale=3):
            chat = _chatbot()
            msg = gr.Textbox(placeholder="e.g. I want eat something leh", show_label=False, autofocus=True)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                reset_btn = gr.Button("Reset")
        with gr.Column(scale=2):
            with gr.Accordion("Debug: latest turn (state + action)", open=True):
                debug = gr.JSON(label="debug")

    msg.submit(respond, [msg, chat, session], [msg, chat, session, debug])
    send.click(respond, [msg, chat, session], [msg, chat, session, debug])
    reset_btn.click(reset, [session], [chat, session, debug])


if __name__ == "__main__":
    demo.launch()
