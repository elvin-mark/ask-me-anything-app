import requests
from bs4 import BeautifulSoup
import torch
from scipy.signal import resample
import numpy as np

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
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def read_and_split_paragraphs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into paragraphs
    paragraphs = content.split('\n')

    # Remove any leading/trailing whitespace from each paragraph
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    return paragraphs

def resample_audio(audio_data, original_rate, new_rate):
    # Convert int16 to float64
    audio_data_float64 = audio_data.astype(np.float64) / 32768.0

    # Determine the duration in seconds
    duration = len(audio_data_float64) / original_rate
    
    # Calculate the number of samples for the new rate
    new_num_samples = int(duration * new_rate)
    
    # Resample the audio data
    resampled_audio_data = resample(audio_data_float64, new_num_samples)
    
    return resampled_audio_data, new_rate