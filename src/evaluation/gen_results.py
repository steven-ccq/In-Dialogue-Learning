import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def format_llama(system, sents):
    prompt = """<s>[INST] <<SYS>>
{}
<</SYS>>
"""
    prompt = prompt.format(system)
    for i in range(len(sents)):
        cur_sent = sents[i]
        if i % 2 == 0:
            cur_sent += ' [/INST] '
        else:
            cur_sent += ' </s><s>[INST] '
        prompt += cur_sent
    prompt = prompt[:-1]
    return prompt

def gen_valid_data(src, dst):
    with open(src, 'r', encoding='utf-8') as f:
        valid_dataset = [json.loads(_) for _ in f]
    for data in valid_dataset:
        profile = ''
        history = data['history']
        prompt = format_llama(profile, history)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        try:
            generate_ids = model.generate(inputs.input_ids[:2048], max_new_tokens=1024)
            resp = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            resp = resp.split('[/INST]')[-1].strip()
            data['llama_response'] = resp
            with open(dst, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False))
                f.write('\n')
        except Exception as e:
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/ConvAI2_DPOC')
    parser.add_argument('--src', default='dataset/ConvAI2/valid_ConvAI2.json')
    parser.add_argument('--dst', default='results/ConvAI2/valid_ConvAI2.json')

    args = parser.parse_args()

    model_path = args.model
    model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    src = args.src
    dst = args.dst
    gen_valid_data(src, dst)