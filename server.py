from flask import Flask,request
from core import AskMeAnything, Config
from argparse import ArgumentParser, BooleanOptionalAction

parser = ArgumentParser()

parser.add_argument("--gpu",type=bool,action=BooleanOptionalAction,help="flag to use gpu or not")
parser.add_argument("--use_memory",type=bool,action=BooleanOptionalAction,help="use memory")

args = parser.parse_args()

model = AskMeAnything(Config(gpu=args.gpu,use_memory=args.use_memory))

app = Flask(__name__)

@app.route("/document",methods=["POST"])
def load_document():
    data = request.get_json()

    upload_type = data.get("type")

    if upload_type == "url":
        url = data.get("url")
        model.load_context_from_wiki(url)
        return {"message":"Document loaded. You can now ask questions."}
    
    if upload_type == "file":
        file = request.files[0]
        model.load_context_from_txt(file.name)
        return {"message":"Document loaded. You can now ask questions."}
    
    if upload_type == "raw_text":
        raw_text = data.get("text")
        model.load_context_from_raw_text(raw_text)
        return {"message":"Document loaded. You can now ask questions."}
    
    return {"message":"No context was loaded"}

@app.route("/ask",methods=["GET"])
def ask():
    data = request.get_json()

    question = data.get("question")
    best = data.get("best",False)

    answers = model.answer(question, best)

    return {"answers":answers}

@app.route("/transcript",methods=["GET"])
def transcript():
    if audio is not None and question == "":
        sampling_rate, audio_arr = audio
        question = model.transcript(audio_arr,sampling_rate)


@app.route("/speech",methods=["GET"])
def speech():
    data = request.get_json()
    script = data["script"]
    
    model.speech(script)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)