from flask import Flask, request, Response, stream_with_context
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation.utils import GenerationConfig
from threading import Thread
import json
import time

app = Flask(__name__)

seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
model_path = '/data/share/Model/baichuan2-13b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_path)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

@app.route('/api/baichuan2_stream', methods=['POST'])
def sample_api():
    data = request.get_json()
    messages = data['prompt']
    # prompt格式为：[{role:,content:}]
    def generate():
        with torch.no_grad():
            streamer = model.chat(tokenizer = tokenizer, messages = messages, stream=True)
        # generated_text = f""
        for new_text in streamer:
            # generated_text += new_text
            response = {
                'status': 'success',
                'result': new_text
            }
            print(response)
            yield (json.dumps(response) + "\0").encode("utf-8")
            time.sleep(0.1)
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False)