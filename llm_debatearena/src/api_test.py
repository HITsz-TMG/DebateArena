import requests
import os
import json
from typing import Iterable, List


# os.environ["https_proxy"] = "http://10.249.43.207:7890"

def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["result"]
            yield output

def vicuna_test():
    print('vicuna')
    url = 'http://219.223.251.156:5000/api/vicuna_stream'
    prompt_string = f"Who are you?"
    data = {'prompt': prompt_string}
    
    gen = get_streaming_response(requests.post(url, json=data, stream=True))

    while True:
        stop = True
        try:
            action = next(gen)
            print(action)
            stop = False
        except StopIteration:
            pass
        if stop:
            break
            
    print()
        
def baichuan2_test():
    print('baichuan2')
    url = 'http://219.223.251.156:5003/api/baichuan2_stream'
    prompt_string = f"Who are you?"
    prompt = [
        {
            'role':'user',
            'content':prompt_string
        }
    ]
    data = {'prompt': prompt}
    
    gen = get_streaming_response(requests.post(url, json=data, stream=True))

    while True:
        stop = True
        try:
            action = next(gen)
            print(action)
            stop = False
        except StopIteration:
            pass
        if stop:
            break
            
    print()
        

def llama2_test():
    print('llama2')
    url = 'http://219.223.251.156:5001/api/llama2_stream'
    prompt_string = f"Who are you?"
    prompt = [
        {
            'role':'user',
            'content':prompt_string
        }
    ]
    data = {'prompt': prompt}
    
    gen = get_streaming_response(requests.post(url, json=data, stream=True))

    while True:
        stop = True
        try:
            action = next(gen)
            print(action)
            stop = False
        except StopIteration:
            pass
        if stop:
            break
            
    print()    
    
def openchat_test():
    print('openchat')
    url = 'http://219.223.251.156:5002/api/openchat_stream'
    prompt_string = f"Who are you?"
    
    data = {'prompt': prompt_string}
    
    gen = get_streaming_response(requests.post(url, json=data, stream=True))

    while True:
        stop = True
        try:
            action = next(gen)
            print(action)
            stop = False
        except StopIteration:
            pass
        if stop:
            break
            
    print()    
            
vicuna_test()