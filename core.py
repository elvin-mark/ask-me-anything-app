from typing import List, Union

from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
import torch
import torch.nn.functional as F

from utils import mean_pooling, get_wikipedia_text

class AskMeAnything:
    def __init__(self):
        self.qa_model_name = "deepset/roberta-base-squad2"
        self.emb_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
        self.emb_model = AutoModel.from_pretrained(self.emb_model_name)
        self.emb_tokenizer = AutoTokenizer.from_pretrained(self.emb_model_name)

    def answer(self, question: str, context: str) -> str:
        inputs = self.qa_tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            answer = self.qa_tokenizer.decode(inputs.input_ids[0, answer_start:answer_end])
        return answer

    def embed(self, text: Union[str, List[str]]) -> torch.tensor:
        inputs = self.emb_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.emb_model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def load_context_from_wiki(self, url: str):
        # self.context = get_wikipedia_text(url).split('.')
        self.context = get_wikipedia_text(url)
        self.context_embeddings = self.embed(self.context)

    def answer_from_context(self, question: str) -> str:
        emb_question = self.embed(question)
        possible_paragraphs = torch.topk(self.context_embeddings @ emb_question.T,k=5,axis=0)
        possible_paragraphs = possible_paragraphs.indices.tolist()
        possible_ans = []
        for idx in possible_paragraphs:
            possible_ans.append(self.answer(question, self.context[idx[0]]))
        return possible_ans
        # idx = torch.argmax(self.context_embeddings @ emb_question.T).item()
        # return self.answer(question, self.context[idx])