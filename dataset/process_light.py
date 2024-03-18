import json
import re
import random

def get_dataset(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        data_list = [_.strip() for _ in f.readlines()]
    dials = []
    user = ''
    bot = ''
    bot_persona = ''
    cur_dial = []
    for data in data_list:
        data = data.replace('labels:', '\\nlabels:')
        if data.startswith('text:_task_speech'):
            user = user.replace('a ', '')
            user = user.replace('the ', '')
            bot = bot.replace('a ', '')
            bot = bot.replace('the ', '')
            dials.append((user, bot, bot_persona, cur_dial[:-1]))
            cur_dial = []
            user = re.findall(r'(?<=_partner_name).*?(?=\\n)', data)[0].strip()
            bot = re.findall(r'(?<=_self_name).*?(?=\\n)', data)[0].strip()
            bot_persona = re.findall(r'(?<=_self_persona).*?(?=\\n)', data)[0].strip()
            try:
                user_sent = re.findall(r'(?<=_partner_say).*?(?=\\n)', data)[0].strip()
                cur_dial.append({'from': 'user', 'value': user_sent})
            except Exception as e:
                continue
        else:
            bot_sent = re.findall(r'(?<=_self_say).*?(?=\\n)', data)[0].strip()
            user_sent = re.findall(r'(?<=_partner_say).*?(?=\\n)', data)[0].strip()
            if len(cur_dial) > 0:
                cur_dial.append({'from': 'bot', 'value': bot_sent})
            cur_dial.append({'from': 'user', 'value': user_sent})
    npc_nums = {}
    for dial in dials:
        if dial[1] not in npc_nums:
            npc_nums[dial[1]] = 0
        npc_nums[dial[1]] += 1
    dataset = {}
    for dial in dials:
        user = dial[0]
        bot = dial[1]
        bot_persona = dial[2]
        content = dial[3]
        if npc_nums[bot] == 1:
            continue
        if (bot, bot_persona) not in dataset:
            dataset[(bot, bot_persona)] = []
        if len(content) < 4:
            continue
        dataset[(bot, bot_persona)].append(content)
    for k, v in dataset.items():
        bot = k[0]
        bot_persona = k[1]
        if len(v) >= 2:
            with open(output_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'role': bot, 'persona': bot_persona, 'dials': v}, ensure_ascii=False))
                f.write('\n')

def get_valid_dataset(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(_) for _ in f]

    for data in data_list:
        profile = data['persona']
        history = data['dials'][:-1]
        if len(history) >= 5:
            history = random.sample(history, 5)
        current = data['dials'][-1]
        history = [sent['value'] for dial in history for sent in dial]
        for sent in current:
            if sent['from'] == 'user':
                history.append(sent['value'])
            else:
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'profile': profile, 'history': history, 'response': sent['value']}))
                    f.write('\n')
                history.append(sent['value'])

if __name__ == '__main__':
    txt_path = './light_dialogue/light_dialogue/speech_test.txt'
    dataset_path = './light_dialogue/light_dialogue.json'
    valid_dataset_path = './light_dialogue/light_dialogue_valid.json'
    get_dataset(txt_path, dataset_path)
    get_valid_dataset(dataset_path, valid_dataset_path)