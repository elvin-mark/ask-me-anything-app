from typing import List, Union

from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, WhisperProcessor, WhisperForConditionalGeneration,VitsModel
import torch
import torch.nn.functional as F

from utils import mean_pooling, get_wikipedia_text, read_and_split_paragraphs,resample_audio,save_audio

class Chatbot:
    def __init__(self):
        self.qa_model_name = "deepset/roberta-base-squad2"
        self.emb_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            self.qa_model_name)
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)

        self.emb_model = AutoModel.from_pretrained(self.emb_model_name)
        self.emb_tokenizer = AutoTokenizer.from_pretrained(self.emb_model_name)
    
    def answer_from_context(self, question: str, context: str) -> str:
        inputs = self.qa_tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            answer = self.qa_tokenizer.decode(
                inputs.input_ids[0, answer_start:answer_end])
        return answer

    def embed(self, text: Union[str, List[str]]) -> torch.tensor:
        inputs = self.emb_tokenizer(
            text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.emb_model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def load_context_from_wiki(self, url: str):
        # self.context = get_wikipedia_text(url).split('.')
        self.context = get_wikipedia_text(url)
        self.context_embeddings = self.embed(self.context)

    def load_context_from_txt(self, path: str):
        self.context = read_and_split_paragraphs(path)
        self.context_embeddings = self.embed(self.context)

    def answer(self, question: str, best=False) -> str:
        # Embed the question
        emb_question = self.embed(question)

        # Find all possible paragraphs that are related to the question
        possible_paragraphs = torch.topk(self.context_embeddings @ emb_question.T, k=5, axis=0)
        possible_paragraphs = possible_paragraphs.indices.tolist()
        
        # For each possible paragrah we find the possible answer
        possible_ans = []
        for idx in possible_paragraphs:
            possible_ans.append(self.answer_from_context(question, self.context[idx[0]]))
        
        # If best flag is True return just the best answer
        if best:
            # idx = torch.argmax(self.context_embeddings @ emb_question.T).item()
            # return self.answer_from_context(question, self.context[idx])
            for ans in possible_ans:
                if len(ans) > 0 and ans != '<s>' and ans != ' ':
                    return ans
            # Default to the one with the highest probability
            return possible_ans[0]

        # If best flag is set to False then give the 5 best answers
        return possible_ans

class AskMeAnything(Chatbot):
    def __init__(self):
        super(AskMeAnything,self).__init__()
        self.asr_model_name = 'openai/whisper-tiny.en'
        self.tts_model_name = "facebook/mms-tts-eng"

        self.asr_processor = WhisperProcessor.from_pretrained(self.asr_model_name)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(self.asr_model_name)

        self.tts_model = VitsModel.from_pretrained(self.tts_model_name)
        self.tts_tokenizer = AutoTokenizer.from_pretrained(self.tts_model_name)

    def transcript(self, audio_arr, sampling_rate: int):
        resample_audio_arr, _  = resample_audio(audio_arr,sampling_rate, 16000)
        input_features = self.asr_processor(resample_audio_arr, sampling_rate=16000, return_tensors="pt").input_features 
        predicted_ids = self.asr_model.generate(input_features)
        transcription = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        if len(transcription) > 0:
            return transcription[0]
        return ""
    
    def speech(self,text:str):
        inputs = self.tts_tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.tts_model(**inputs).waveform
        save_audio(data=output.float().numpy(),rate=self.tts_model.config.sampling_rate)
