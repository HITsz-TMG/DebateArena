import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from flask import Flask, request, Response, stream_with_context
from transformers import TextIteratorStreamer
from fastchat.model import load_model, get_conversation_template, add_model_args
import argparse
from threading import Thread
import json
from time import sleep

app = Flask(__name__)

parser = argparse.ArgumentParser()
add_model_args(parser)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--message", type=str, default="Hello! Who are you?")
args = parser.parse_args()
args.model_path = '/data/share/vicuna-13b-v1.5'

if "t5" in args.model_path and args.repetition_penalty == 1.0:
    args.repetition_penalty = 1.2
    
    
model, tokenizer = load_model(
    args.model_path,
    device=args.device,
    num_gpus=args.num_gpus,
    max_gpu_memory=args.max_gpu_memory,
    load_8bit=args.load_8bit,
    cpu_offloading=args.cpu_offloading,
    debug=args.debug,
)


@app.route('/api/vicuna_stream', methods=['POST'])
def sample_api():

    data = request.get_json()
    prompt_text = data['prompt']
    
    def generate():
        msg = prompt_text
        conv = get_conversation_template(args.model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Run inference
        inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs, streamer=streamer, 
                            max_new_tokens=args.max_new_tokens, 
                            temperature=args.temperature,
                            repetition_penalty=args.repetition_penalty,
                            do_sample=True if args.temperature > 1e-5 else False)
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
            sleep(0.1)
            
            
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)

    
    
    

    
    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)