from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import VectorDBQA
import gradio as gr
import os
import time
import openai
import ingest

chat = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.9, 
                #   streaming=True
                  )

vectorstore = FAISS.load_local(
        folder_path="faiss", 
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )

def create_generator(history):
    user_input = history[-1][0]

    if not isinstance(user_input, str):
        return "업로드가 완료되었습니다."

    similar_docs = vectorstore.similarity_search(user_input, k=3)
    
    context = "\n\n".join([doc.page_content for doc in similar_docs])
    prompt = f"""
    다음은 사용자가 입력한 질문입니다:
    "{user_input}"
    
    그리고 관련 문서입니다:
    "{context}"
    
    질문이 문서와 관련이 있다면 관련 문서를 참조해 대답해주세요.
    """
    
    gpt_response = chat.predict(prompt)
    return gpt_response

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

def add_file(history, file):
    ingest.pdfToVectorStore(file)
    history = history + [((file.name,), None)]
    return history

def bot(history):
    response = create_generator(history)
    history[-1][1] = response
    return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join("./avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="텍스트를 입력하고 엔터를 치거나 이미지를 업로드하세요",
            container=False,
        )
        btn = gr.UploadButton("Upload", file_types=[".pdf"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
demo.launch(server_name='0.0.0.0')