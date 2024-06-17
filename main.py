import gradio as gr
from core import AskMeAnything

model = AskMeAnything()


def load_document(file, url):
    if url is not None and url != "":
        model.load_context_from_wiki(url)
        return "Document loaded. You can now ask questions."
    if file is not None:
        model.load_context_from_txt(file.name)
        return "Document loaded. You can now ask questions."
    return "No context was loaded"
# Function to generate answers based on the document


def ask_question(question, audio, best):
    if question is not None and question != "":
        answers = model.answer_from_context(question, best)
        if best:
            return question, answers
        else:
            return question, "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
    if audio is not None:
        sampling_rate, audio_arr = audio
        question = model.transcript(audio_arr,sampling_rate)
        answers = model.answer_from_context(question, best)
        if best:
            return question, answers
        else:
            return question, "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
    return "", ""

# Create the Gradio interface
file_input = gr.File(label="Upload Document")
ulr_input = gr.Textbox(
    lines=1, placeholder="Paste the Wikipedia URL here...", label="Wikipedia URL")
text_input = gr.Textbox(
    lines=2, placeholder="Enter your question here...", label="Your Question")
audio_input = gr.Audio(sources=["microphone"])
best_flag_input = gr.Checkbox(
    label="Best Answer", info="Just give the best answer", value=True)
text_question_ref = gr.Textbox(label="Your question was:")
text_output = gr.Textbox(label="Possible Answers:")

iface = gr.Interface(
    fn=ask_question,
    inputs=[text_input, audio_input, best_flag_input],
    outputs=[text_question_ref,text_output],
    title="Document Q&A Chatbot",
    description="Upload a document and ask questions about it."
)

# Add the document upload interface
iface_upload = gr.Interface(
    fn=load_document,
    inputs=[file_input, ulr_input],
    outputs="text",
    title="Upload Document",
    description="Upload a document to ask questions about it."
)

# Combine both interfaces
app = gr.TabbedInterface([iface_upload, iface], [
                         "Upload Document", "Ask Questions"])

app.launch(server_name="0.0.0.0", server_port=8080)
