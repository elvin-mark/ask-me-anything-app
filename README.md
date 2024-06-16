# ask-me-anything-app

Ask Me Anything

# Run Me

## Locally

### Prerequisites

```sh
pip install -r requirements.txt
```

### Running

```sh
python main.py
```

## Docker

### Building

```sh
docker build -t tools/ask-me-anything .
```

### Running

```sh
docker run -it --name ask-me-anything-app -p 8080:8080 tools/ask-me-anything
```
