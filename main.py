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


def ask_question(question, audio):
    print(question, audio)
    if question is not None and question != "":
        answers = model.answer_from_context(question)
        return "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
    if audio is not None:
        pass
        return "No answers"


# Create the Gradio interface
file_input = gr.File(label="Upload Document")
ulr_input = gr.Textbox(
    lines=1, placeholder="Paste the Wikipedia URL here...", label="Wikipedia URL")
text_input = gr.Textbox(
    lines=2, placeholder="Enter your question here...", label="Your Question")
audio_input = gr.Audio(sources=["microphone"])
text_output = gr.Textbox(label="Possible Answers:")

iface = gr.Interface(
    fn=ask_question,
    inputs=[text_input, audio_input],
    outputs=text_output,
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

app.launch()
