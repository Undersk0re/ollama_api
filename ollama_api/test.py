from client import OllamaClient
from pydantic import BaseModel

class testSchema(BaseModel):
    lista: list[str,str]
    diz : dict

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

    # Test using_custom_schema
    print("\nTESTING using_custom_schema")
    try:
        schema_resp = client.using_custom_schema(
            user_prompt="List 5 key components of a computer's hardware",
            system_prompt="You are a tech analyst. Respond only with valid JSON matching the schema.",
            output_format="json",
            schema=testSchema.model_json_schema,
            model='llama3.2:1b'  # Explicitly add model parameter
        )
        print("Schema Response:", schema_resp)
    except Exception as e:
        print("Schema Error:", str(e))
    print("-"*50)
