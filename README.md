# ask-me-anything-app

# About

Ask Me Anything App is an app aiming to be a simple AI assistant to help you find answers through multiple source materials. It allows you to upload files or send a URL to extract the text from that page. After you have upload you source materials, you can start asking questions that may be found in those source materials. You can ask by typing your question or by using the speech recognition functionality to just speak your question. Then you will receive the answer via text and an automatic generated speech.

# Models being used

Here are all the models that are being used by this application. 

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): used for indexing the paragraphs.
- [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2): used to find the answer within the paragraph
- [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en): used to converse with the chatbot
- [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng): used to vocalize the answers

# Run Me

## Locally

### Prerequisites

You will need ptyhon > 3.9 and pip install on your local machine. Then you can run the following command to install all the required packages.

```sh
pip install -r requirements.txt
```

### Running

After installing all the dependencies in the previous step, you can start the server by running the following commnad. Then you can access to the application through the following link: `localhost:8080`

```sh
python main.py
```

## Docker

### Building

To avoid any issues with your current environment, you can opt for running the application in a docker container. First, run the following command to build the docker image.

```sh
docker build -t tools/ask-me-anything .
```

Then run the following command to start the docker container. Then you can access to the application through the following link: `localhost:8080`

### Running

```sh
docker run -it --name ask-me-anything-app -p 8080:8080 tools/ask-me-anything
```
