"""_summary_


Python client for interacting with an Ollama server, providing a high-level interface to manage and utilize language models.
Offers methods for text generation, chat completion, embeddings creation, and model management (create/pull/push/delete).
Includes tested core functionalities for completions, chat, embeddings, and model listing, with additional untested model operations.
Default connection is set to 'http://localhost:11434' unless customized during initialization.


The `OllamaClient` class provides a Python interface to interact with the Ollama API, enabling users to manage and utilize machine 
learning models for tasks such as text generation, chat completion, embeddings, and model lifecycle operations. Key features include:  

- API Abstraction: Simplifies HTTP requests to Ollama endpoints (e.g., `generate`, `chat`, `embeddings`) via helper methods.  
- Model Management: Supports model creation, deletion, copying, pulling/pushing (with registry), and metadata inspection.  
- Error Handling: Wraps API calls with try-except blocks to return structured error messages.  
- Configuration: Uses constants for default host (`localhost`), connection settings, and endpoint mappings.  
- Stream Support: Many methods include a `stream` parameter for handling real-time responses (though streaming logic is not fully implemented in tested methods).  

Tested methods include text generation (`request_completion`), chat interactions (`request_chat_completion`), embeddings (`generate_embeddings`), 
and model listing (`list_local_models`, `list_running_models`). Untested methods cover advanced model operations (e.g., `copy_model`, `delete_model`). 
The class is designed for flexibility, allowing customization of host targets and optional parameters via `**kwargs`.

Simplicity is not an option

"""


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
            'embeddings': "{'model': model, 'prompt': prompt, **kwargs}",
            'schema' : """{'model': model, 'stream': stream,'keep_alive': 10, 
                'system': system_prompt, 'prompt': user_prompt,"format": schema() , **kwargs}"""
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

    def _build_rest(self, key) -> tuple[dict,str]:
        return ( self.CONST['data'][key],
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

    # tested methods
    def request_completion(self, model: str, prompt: str, stream: bool = False, **kwargs) -> tuple:
        """Tested"""
        data_str, url = self._build_rest('generate')
        data = eval(data_str)
        full = self._try_req('post', url, data)
        response = full['response']
        emb_contex = ['contex'] # differ from normal embeddings
        return response, emb_contex

    def request_chat_completion(self, model: str, messages: list[dict], stream: bool = False, **kwargs)-> str:
        """Tested"""
        data_str, url = self._build_rest('chat')
        data = eval(data_str)
        return self._try_req('post', url, data)['message']['content']

    def generate_embeddings(self, model: str, prompt: str, **kwargs):
        """Tested"""
        data_str, url = self._build_rest('embeddings')
        data = eval(data_str)
        return self._try_req('post', url, data)['embedding']

    def list_local_models(self) -> dict:
        """Tested"""
        url = self.base_url + self.CONST['endpoint_exceptions']['tags']
        return self._try_req('get', url)

    def list_running_models(self) -> dict:
        """Tested"""
        url = self.base_url + self.CONST['endpoint_exceptions']['running']
        return self._try_req('get', url)
    
    def using_custom_schema( self,model,user_prompt: str, system_prompt: str = "You are a helpful assistant", 
                                schema:type=None , stream=False  ,**kwargs) -> str: 
        response, _ = self.request_completion( model=model, prompt=user_prompt, system=system_prompt,
                        format=schema() if schema else None, stream=stream,  **kwargs )
        return response




    
    # untested methods
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

