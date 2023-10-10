from flask import Flask, request, jsonify, stream_with_context, Response
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, TextIteratorStreamer
from threading import Thread
from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
import argparse
import json
import time
parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--message", type=str, default="Hello! Who are you?")
args = parser.parse_args()

app = Flask(__name__)

seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
model_path = 'PATH_TO_openchat_v3.2_super'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        return_dict=True,
        load_in_8bit=False,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

def prompt_format(prompt):
    head = "GPT4 User: "
    tail = "<|end_of_turn|>GPT4 Assistant:"
    return f"{head}{prompt}{tail}"

@app.route('/api/openchat_stream', methods=['POST'])
def sample_api():
    data = request.get_json()
    # data格式为：[{role:,content:}]
    prompt = data['prompt']
    prompt = prompt_format(prompt)
    chat = tokenizer(prompt).input_ids
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    def generate():
        with torch.no_grad():
            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            tokens= tokens.to("cuda")
            generation_kwargs = dict(input_ids=tokens, 
                                    streamer=streamer, 
                                    max_new_tokens=args.max_new_tokens, 
                                    temperature=args.temperature,
                                    repetition_penalty=args.repetition_penalty,
                                    do_sample=True if args.temperature > 1e-5 else False,
                                    top_p=args.top_p,
                                    use_cache=True,
                                    top_k=args.top_k,
                                    eos_token_id = [tokenizer.eos_token_id])
            Thread(target=model.generate, kwargs=generation_kwargs).start()
            generated_text = f""
            for new_text in streamer:
                generated_text += new_text
                response = {
                    'status': 'success',
                    'result': generated_text
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
    app.run(host='0.0.0.0', port=5002, debug=False)