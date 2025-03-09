import requests

class OllamaClient:
    CONST = {
        "data": {
            'generate': "{'model': {model}, 'prompt': {prompt}, 'stream': {stream}, **{kwargs}}",
            'chat': "{'model': {model}, 'messages': {messages}, 'stream': {stream}, **{kwargs}}",
            'create': "{'name': {name}, 'modelfile': {modelfile}, 'stream': {stream}}",
            'pull': "{'name': {name}, 'insecure': {insecure}, 'stream': {stream}}",
            'push': "{'name': {name}, 'insecure': {insecure}, 'stream': {stream}}",
            'show': "{'name': {name}, 'verbose': {verbose}}",
            'copy': "{'source': {source}, 'destination': {destination}}",
            'delete': "{'name': {name}}",
            'embeddings': "{'model': {model}, 'prompt': {prompt}, **{kwargs}}"
        },
        'host': 'localhost', # <------------------ insert your Ollama host here
        'connection': "http://{host}:11434",
        'endpoint_exceptions': {
            'tags': '/api/tags',
            'running': '/api/ps'
        }
    }

    def __init__(self, target_hostname: str = CONST['host']) -> None:
        self.base_url = self.CONST['connection'].format(host=target_hostname)

    def __build_data(self, template_key: str, formatted_params: dict) -> dict:
        data_str = self.CONST['data'][template_key].format(**formatted_params)
        return dict(eval(data_str))

    def __build_endpoint(self, key: str) -> str:
        return self.CONST['endpoint_exceptions'].get(key, f"/api/{key}")

    def _build_rest(self, template_key: str, **params) -> tuple:
        formatted_params = {k: repr(v) for k, v in params.items()}
        return (
            self.__build_data(template_key, formatted_params),
            self.base_url + self.__build_endpoint(template_key)
        )

    def request_completion(self, model, prompt, stream=True, **kwargs):
        data, url = self._build_rest('generate', model=model, prompt=prompt, stream=stream, kwargs=kwargs)
        return requests.post(url, json=data)

    def request_chat_completion(self, model, messages, stream=True, **kwargs):
        data, url = self._build_rest('chat', model=model, messages=messages, stream=stream, kwargs=kwargs)
        return requests.post(url, json=data)

    def request_model(self, name, modelfile, stream=True):
        data, url = self._build_rest('create', name=name, modelfile=modelfile, stream=stream)
        return requests.post(url, json=data)

    def request_pull_model(self, name, insecure=False, stream=True):
        data, url = self._build_rest('pull', name=name, insecure=insecure, stream=stream)
        return requests.post(url, json=data)

    def request_push_model(self, name, insecure=False, stream=True):
        data, url = self._build_rest('push', name=name, insecure=insecure, stream=stream)
        return requests.post(url, json=data)

    def show_model_information(self, name, verbose=False):
        data, url = self._build_rest('show', name=name, verbose=verbose)
        return requests.post(url, json=data).json()

    def copy_model(self, source, destination):
        data, url = self._build_rest('copy', source=source, destination=destination)
        return requests.post(url, json=data).json()

    def delete_model(self, name):
        data, url = self._build_rest('delete', name=name)
        return requests.delete(url, json=data).json()

    def generate_embeddings(self, model, prompt, **kwargs):
        data, url = self._build_rest('embeddings', model=model, prompt=prompt, kwargs=kwargs)
        return requests.post(url, json=data).json()

    def list_local_models(self):
        return requests.get(self.base_url + self.CONST['endpoint_exceptions']['tags']).json()

    def list_running_models(self):
        return requests.get(self.base_url + self.CONST['endpoint_exceptions']['running']).json()