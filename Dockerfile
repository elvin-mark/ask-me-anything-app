FROM python:3.10-alpine

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt --break-system-packages

EXPOSE 8080

ENTRYPOINT [ "python", "main.py" ]