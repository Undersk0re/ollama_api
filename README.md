# Ollama API Python Client

A Python client for interacting with the Ollama API.

## Installation

```bash
pip install ollama_api
```

## Usage

```python
from ollama_api import OllamaClient

client = OllamaClient()
response = client.generate_completion(model="llama3", prompt="Why is the sky blue?")
print(response)
```

## Documentation

For more details, refer to the [Ollama API documentation](https://github.com/ollama/ollama/tree/main/docs).


### Step 6: Install and Test

Navigate to the root directory of your package and install it locally:

```bash
# build locally
python setup.py sdist bdist_wheel
```

```bash
pip install .
```

Now you can test the package by importing and using the `OllamaClient` class in a Python script or interactive session.

This basic package structure and client class should give you a good starting point for interacting with the Ollama API using Python. You can expand and refine it further based on your specific needs and the API's capabilities.
