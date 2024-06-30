import gradio as gr
from core import AskMeAnything, Config
from argparse import ArgumentParser, BooleanOptionalAction

parser = ArgumentParser()

parser.add_argument("--gpu",type=bool,action=BooleanOptionalAction,help="flag to use gpu or not")
parser.add_argument("--use_memory",type=bool,action=BooleanOptionalAction,help="use memory")
parser.add_argument("--share",type=bool,action=BooleanOptionalAction,help="share gradio link")

args = parser.parse_args()


model = AskMeAnything(Config(gpu=args.gpu,use_memory=args.use_memory))


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
    answers = ""
    if audio is not None and question == "":
        sampling_rate, audio_arr = audio
        question = model.transcript(audio_arr,sampling_rate)

    answers = model.answer(question, best)
    
    if not best:
        answers = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
    
    model.speech(answers)

    return question, answers, "tmp.wav"

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
audio_output =gr.Audio()

iface = gr.Interface(
    fn=ask_question,
    inputs=[text_input, audio_input, best_flag_input],
    outputs=[text_question_ref,text_output,audio_output],
    examples=[["What is physics?"],["What is mathematics?"],["What is chemistry?"]],
    title="Document Q&A Chatbot",
    description="Upload a document and ask questions about it."
)

# Add the document upload interface
iface_upload = gr.Interface(
    fn=load_document,
    inputs=[file_input, ulr_input],
    outputs="text",
    examples=[[None,"https://en.wikipedia.org/wiki/Physics"],[None,"https://en.wikipedia.org/wiki/Mathematics"],[None,"https://en.wikipedia.org/wiki/Chemistry"]],
    title="Upload Document",
    description="Upload a document to ask questions about it."
)

# Combine both interfaces
app = gr.TabbedInterface([iface_upload, iface], [
                         "Upload Document", "Ask Questions"])

app.launch(server_name="0.0.0.0", server_port=8080,share=args.share)
