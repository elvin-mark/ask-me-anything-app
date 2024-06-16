import requests
from bs4 import BeautifulSoup
import torch

def get_wikipedia_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all paragraphs
    paragraphs = soup.find_all('p')

    # Extract text from paragraphs
    text = []
    for paragraph in paragraphs:
        text.append(paragraph.get_text())

    # Join the text content
    # return '\n'.join(text)

    # Return all paragraphs
    return text

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)