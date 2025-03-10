import requests

class OllamaClient:
    CONST = {
        "data": {
            'generate': "{'model': model, 'prompt': prompt, 'stream': stream, **kwargs}",
            'chat': "{'model': model, 'messages': messages, 'stream': stream, **kwargs}",
            'create': "{'name': name, 'modelfile': modelfile, 'stream': stream}",
            'pull': "{'name': name, 'insecure': insecure, 'stream': stream}",
            'push': "{'name': name, 'insecure': insecure, 'stream': stream}",
            'show': "{'name': name, 'verbose': verbose}",
            'copy': "{'source': source, 'destination': destination}",
            'delete': "{'name': name}",
            'embeddings': "{'model': model, 'prompt': prompt, **kwargs}"
        },
        'host': 'localhost',
        'connection': "http://{host}:11434",
        'endpoint_exceptions': {
            'tags': '/api/tags',
            'running': '/api/ps'
        }
    }

    def __init__(self, target_hostname: str = CONST['host']) -> None:
        self.base_url = self.CONST['connection'].format(host=target_hostname)

    def _build_rest(self, key) -> tuple:
        return (
            self.CONST['data'][key],
            self.base_url + self.CONST['endpoint_exceptions'].get(key, f"/api/{key}")
        )

    def _try_req(self, method: str, url: str, json_data: dict = None) -> dict:
        try:
            if method == 'post':
                response = requests.post(url, json=json_data)
            elif method == 'get':
                response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}

    def request_completion(self, model, prompt, stream=False, **kwargs) -> tuple:
        data_str, url = self._build_rest('generate')
        data = eval(data_str)
        full = self._try_req('post', url, data)
        response = full['response']
        emb_contex = ['contex'] # differ from normal embeddings
        return response, emb_contex

    def request_chat_completion(self, model, messages, stream=False, **kwargs):
        data_str, url = self._build_rest('chat')
        data = eval(data_str)
        return self._try_req('post', url, data)['message']['content']
    
    def request_model(self, name, modelfile, stream=False):
        data_str, url = self._build_rest('create')
        data = eval(data_str)
        return self._try_req('post', url, data)

    def request_pull_model(self, name, insecure=False, stream=False):
        data_str, url = self._build_rest('pull')
        data = eval(data_str)
        return self._try_req('post', url, data)

    def request_push_model(self, name, insecure=False, stream=False):
        data_str, url = self._build_rest('push')
        data = eval(data_str)
        return self._try_req('post', url, data)

    def show_model_information(self, name, verbose=False):
        data_str, url = self._build_rest('show')
        data = eval(data_str)
        return self._try_req('post', url, data)

    def copy_model(self, source, destination):
        data_str, url = self._build_rest('copy')
        data = eval(data_str)
        return self._try_req('post', url, data)

    def delete_model(self, name):
        data_str, url = self._build_rest('delete')
        data = eval(data_str)
        return self._try_req('post', url, data)

    def generate_embeddings(self, model, prompt, **kwargs):
        data_str, url = self._build_rest('embeddings')
        data = eval(data_str)
        return self._try_req('post', url, data)['embedding']

    def list_local_models(self):
        url = self.base_url + self.CONST['endpoint_exceptions']['tags']
        return self._try_req('get', url)

    def list_running_models(self):
        url = self.base_url + self.CONST['endpoint_exceptions']['running']
        return self._try_req('get', url)
    

if __name__ == "__main__":
    uri='localhost'
    client = OllamaClient(uri)
    
    # Test request_completion
    print("TESTING request_completion")
    try:
        completion_resp, _ = client.request_completion(
            model='llama3.2:1b',
            prompt="Explain quantum mechanics in simple terms",
            stream=False,
            temperature=0.7
        )
        print("Completion Response:", completion_resp)
    except Exception as e:
        print("Completion Error:", str(e))
    print("-"*50)

    # Test request_chat_completion
    print("\nTESTING request_chat_completion")
    try:
        chat_resp = client.request_chat_completion(
            model='llama3.2:1b',
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Why is the sky blue?"}
            ],
            stream=False
        )
        print("Chat Response:", chat_resp)
    except Exception as e:
        print("Chat Error:", str(e))
    print("-"*50)

    # Test generate_embeddings
    print("\nTESTING generate_embeddings")
    try:
        embedding_resp = client.generate_embeddings(
            model='llama3.2:1b',
            prompt="This is a test sentence for embeddings"
        )
        print("Embeddings Response:", embedding_resp)
    except Exception as e:
        print("Embeddings Error:", str(e))
    print("-"*50)

    # Test list_local_models
    print("\nTESTING list_local_models")
    try:
        local_models = client.list_local_models()
        print("Local Models:", local_models)
    except Exception as e:
        print("Local Models Error:", str(e))
    print("-"*50)

    # Test list_running_models
    print("\nTESTING list_running_models")
    try:
        running_models = client.list_running_models()
        print("Running Models:", running_models)
    except Exception as e:
        print("Running Models Error:", str(e))
    print("-"*50)

