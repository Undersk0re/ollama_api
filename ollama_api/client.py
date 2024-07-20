# ollama_api/client.py
import requests


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def generate_completion(self, model, prompt, stream=True, **kwargs):
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        response = requests.post(url, json=data)
        return response.json()

    def generate_chat_completion(self, model, messages, stream=True, **kwargs):
        url = f"{self.base_url}/api/chat"
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        response = requests.post(url, json=data)
        return response.json()

    def create_model(self, name, modelfile, stream=True):
        url = f"{self.base_url}/api/create"
        data = {
            "name": name,
            "modelfile": modelfile,
            "stream": stream
        }
        response = requests.post(url, json=data)
        return response.json()

    def list_local_models(self):
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        return response.json()

    def show_model_information(self, name, verbose=False):
        url = f"{self.base_url}/api/show"
        data = {
            "name": name,
            "verbose": verbose
        }
        response = requests.post(url, json=data)
        return response.json()

    def copy_model(self, source, destination):
        url = f"{self.base_url}/api/copy"
        data = {
            "source": source,
            "destination": destination
        }
        response = requests.post(url, json=data)
        return response.json()

    def delete_model(self, name):
        url = f"{self.base_url}/api/delete"
        data = {
            "name": name
        }
        response = requests.delete(url, json=data)
        return response.json()

    def pull_model(self, name, insecure=False, stream=True):
        url = f"{self.base_url}/api/pull"
        data = {
            "name": name,
            "insecure": insecure,
            "stream": stream
        }
        response = requests.post(url, json=data)
        return response.json()

    def push_model(self, name, insecure=False, stream=True):
        url = f"{self.base_url}/api/push"
        data = {
            "name": name,
            "insecure": insecure,
            "stream": stream
        }
        response = requests.post(url, json=data)
        return response.json()

    def generate_embeddings(self, model, prompt, **kwargs):
        url = f"{self.base_url}/api/embeddings"
        data = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        response = requests.post(url, json=data)
        return response.json()

    def list_running_models(self):
        url = f"{self.base_url}/api/ps"
        response = requests.get(url)
        return response.json()
