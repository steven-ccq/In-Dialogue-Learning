import json
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy
from spacy.matcher import Matcher
import random
import pickle
import os
import numpy as np
import faiss

encode_model = SentenceTransformer('all-mpnet-base-v2').cuda()

persona_extraction_model = AutoModelForCausalLM.from_pretrained('persona_extraction').half().cuda()
persona_extraction_tokenizer = AutoTokenizer.from_pretrained('persona_extraction')

# convert dataset from txt to json
def convert(input_file_path, output_file_path, data_type='sft'):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data_list = [_.strip() for _ in f.readlines()]
    # SFT
    if data_type == 'sft':
        dataset = []
        cur_dial = {
            'profile': [],
            'dialogue': []
        }
        cur_idx = 0
        for _ in data_list:
            cur_idx = int(_.split(' ')[0])
            _ = ' '.join(_.split(' ')[1:])
            sents = _.split('\t')
            if cur_idx == 1:
                dataset.append(cur_dial.copy())
                cur_dial['profile'] = []
                cur_dial['dialogue'] = []
            if len(sents) == 1:
                assert 'your persona' in sents[0], sents[0]
                cur_dial['profile'].append(sents[0])
            else:
                cur_dial['dialogue'].append((sents[0], sents[1]))
        dataset.append(cur_dial.copy())
    elif data_type == 'dpo':
        dataset = []
        cur_dial = {
            'profile': [],
            'dialogue': []
        }
        cur_idx = 0
        for _ in data_list:
            cur_idx = int(_.split(' ')[0])
            _ = ' '.join(_.split(' ')[1:])
            sents = _.split('\t')
            if cur_idx == 1:
                dataset.append(cur_dial.copy())
                cur_dial['profile'] = []
                cur_dial['dialogue'] = []
            if len(sents) == 1:
                assert 'your persona' in sents[0], sents[0]
                cur_dial['profile'].append(sents[0])
            else:
                cands = sents[-1].split('|')[::-1][1:]
                cur_dial['dialogue'].append((sents[0], sents[1], cands))
        dataset.append(cur_dial.copy())

    with open(output_file_path, 'a', encoding='utf-8') as f:
        for _ in dataset:
            f.write(json.dumps(_, ensure_ascii=False))
            f.write('\n')
# extract training cases from dialogues
def extract(input_file_path, output_file_path, data_type='sft'):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(_) for _ in f]
    # SFT
    if data_type == 'sft':
        with open(output_file_path, 'a', encoding='utf-8') as f:
            for _ in dataset:
                profile = '\n'.join([p.replace('your persona: ', '') for p in _['profile']])
                dials = []
                sents = _['dialogue']
                for user, bot in sents:
                    dials.append({'from': 'user', 'value': user})
                    dials.append({'from': 'bot', 'value': bot})
                f.write(json.dumps({'conversations': {'profile': profile,'dials': dials}},ensure_ascii=False))
                f.write('\n')
    # DPO
    if data_type == 'dpo':
        with open(output_file_path, 'a', encoding='utf-8') as f:
            for _ in dataset:
                profile = '\n'.join([p.replace('your persona: ', '') for p in _['profile']])
                dials = []
                sents = _['dialogue']
                for user, bot, cands in sents:
                    dials.append({'from': 'user', 'value': user})
                    dials.append({'from': 'bot', 'value': bot, 'cands': cands})
                f.write(json.dumps({'conversations': {'profile': profile,'dials': dials}},ensure_ascii=False))
                f.write('\n')
# compute cosine similarity
def compute_cosine_sim(vec1, vec2):
    return F.cosine_similarity(vec1, vec2)[0].item()
# extract persona intensive sentence from dialogues
def get_persona_sent(sents):
    dial_str = '\n'.join(sents)
    inputs = persona_extraction_tokenizer(dial_str, return_tensors='pt').to(persona_extraction_model.device)
    generate_ids = persona_extraction_model.generate(inputs.input_ids, max_new_tokens=1024)
    persona_triples = persona_extraction_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(dial_str):].split('\n')
    persona_sents = []
    for triple in persona_triples:
        eles = triple.split(' ')
        if len(eles) < 3:
            continue
        o = ' '.join(triple.split(' ')[2:])
        for sent in sents:
            if o in sent and sent not in persona_sents:
                persona_sents.append(sent)
                break
    if persona_sents:
        return '\n'.join(persona_sents)
    return '\n'.join(sents)
# calculate convED between two dialogues
def dial_rel_score(dial1, dial2, alpha=1):
    dial1_vec = []
    dial2_vec = []
    for sent in dial1:
        speaker = sent['from']
        utterance = sent['value']
        vec = encode_model.encode(utterance)
        dial1_vec.append((speaker, vec))
    for sent in dial2:
        speaker = sent['from']
        utterance = sent['value']
        vec = encode_model.encode(utterance)
        dial2_vec.append((speaker, vec))
    turn1, turn2 = len(dial1_vec), len(dial2_vec)
    dp = [[0] * (turn1+1)] * (turn2+1)
    for i in range(turn1):
        for j in range(turn2):
            if i == 0:
                dp[i][j] = j  
            elif j == 0:
                dp[i][j] = i
            else:
                dp[i][j] = min(1+dp[i][j-1], # insertion
                               1+dp[i-1][j], # deletion
                               alpha * (1-F.cosine_similarity(dial1_vec[i-1][1], dial2_vec[j-1][1])) if dial1_vec[i-1][0] == dial1_vec[j-1][0] else 1e9 + 7 + dp[i-1][j-1]) # substitution
    return dp[turn1][turn2]
# dynamic
def dial_sort(dials):
    length = len(dials)
    dis = []
    for i in range(length):
        dis.append([])
        for j in range(length):
            if i == j:
                dis[i].append(0)
            else:
                rel = dial_rel_score(dials[i], dials[j])
                dis[i].append(rel)
    visited = [False] * length
    path = []
    cur_index = 0
    visited[cur_index] = True
    path.append(dials[cur_index])

    while len(path) < length:
        min_distance = 100
        next_index = None
        for i in range(length):
            if not visited[i] and dis[cur_index][i] < min_distance:
                min_distance = dis[cur_index][i]
                next_index = i
        visited[next_index] = True
        path.append(dials[next_index])
        cur_index = next_index

    return path[1:]
# static
def persona_cluster(clf, sessions):
    embeds = encode_model.encode(sessions)
    ydata = clf.fit_predict(embeds)
    ydata = list(ydata)
    return ydata

def get_msl(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(_)['conversations'] for _ in f]
    grouped_data_list = {}
    for i in range(len(data_list)):
        profile = data_list[i]['profile']
        profile = sorted(profile.split('\n'))
        profile = '\n'.join(profile)
        if profile not in grouped_data_list:
            grouped_data_list[profile] = []
        grouped_data_list[profile].append(data_list[i]['dials'])
    cluster_num = 3
    clf = KMeans(n_clusters=cluster_num)
    for k, v in grouped_data_list.items():
        if len(v) < 2:
            continue
        if len(v) < 2 * cluster_num:
            history = v[:-1]
            dial = v[-1]
            sorted_sessions = dial_sort([dial]+history)
            history = []
            for session in sorted_sessions:
                history += session
            item = {
                'profile': k,
                'history': history,
                'dial': dial
            }
            with open(output_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'conversations': item}, ensure_ascii=False))
                f.write('\n')
        else:
            sessions = []
            for dial in v:
                dial = get_persona_sent([_['value'] for _ in dial if _['from'] == 'bot'])
                sessions.append(dial)
            ydata = persona_cluster(clf, sessions)
            clusters = [[] for i in range(cluster_num)]
            for i in range(len(ydata)):
                clusters[ydata[i]].append(v[i])
            for dials in clusters:
                if len(dials) < 2:
                    continue
                dial = dials[-1]
                history = dials[:-1]
                sorted_sessions = dial_sort([dial]+history)
                history = []
                for session in sorted_sessions:
                    history += session
                item = {
                    'profile': k,
                    'history': history,
                    'dial': dial
                }
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'conversations': item}, ensure_ascii=False))
                    f.write('\n')
# get dataset for DPO
def get_dpo(input_file_path, output_file_path):
    def rm_cands(data_list):
        for i in range(len(data_list)):
            if 'cands' in data_list[i]:
                data_list[i].pop('cands')

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(_)['conversations'] for _ in f]

    grouped_data_list = {}
    for i in range(len(data_list)):
        profile = data_list[i]['profile']
        profile = sorted(profile.split('\n'))
        profile = '\n'.join(profile)
        if profile not in grouped_data_list:
            grouped_data_list[profile] = []
        grouped_data_list[profile].append(data_list[i]['dials'])

    for k, v in grouped_data_list.items():
        if len(v) < 2:
            continue
        dial = v[-1]
        history = []
        for _ in v[:-1]:
            history += _
        rm_cands(history)
        for _ in dial:
            speaker = _['from']
            utterance = _['value']
            if speaker == 'user':
                history.append(_)
                continue
            negs = _['cands']
            item = {
                'profile': k,
                'history': history,
                'pos': utterance,
                'negs': negs
            }
            with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'conversations': item}, ensure_ascii=False))
                    f.write('\n')

            _.pop('cands')
            history.append(_)
# generate embeddings for this corpus
def gen_embs(input_file_path, output_dir):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(_) for _ in f]

    profiles = []
    for data in data_list:
        profile = data['profile']
        profile = [_.replace('your persona: ', '') for _ in profile]
        profile = sorted(profile)
        profile = '\n'.join(profile)
        if profile not in profiles:
            profiles.append(profile)

    profile_idx = {}
    idx_profile = {}
    for idx, profile in enumerate(profiles):
        profile_idx[profile] = idx
        idx_profile[idx] = profile

    with open(os.path.join(output_dir, 'profile_idx.json'), 'w') as f:
        json.dump(profile_idx, f)
    with open(os.path.join(output_dir, 'idx_profile.json'), 'w') as f:
        json.dump(idx_profile, f)
        
    all_embs = []
    emb_meta = []
    idx = 0
    for data in data_list:
        profile = data['profile']
        profile = [_.replace('your persona: ', '') for _ in profile]
        profile = sorted(profile)
        profile = '\n'.join(profile)
        assert profile in profile_idx, profile
        p_idx = profile_idx[profile]
        dials = data['dialogue']
        for dial in dials:
            for i in range(len(dial)):
                sent = dial[i]
                if i % 2 == 0:
                    emb_meta.append({'profile_idx': p_idx, 'role': 'user', 'text': sent})
                else:
                    emb_meta.append({'profile_idx': p_idx, 'role': 'bot', 'text': sent})
                emb = encode_model.encode(sent)
                all_embs.append(emb)
        idx += 1
        if idx % 100 == 0:
            print(idx)

    with open(os.path.join(output_dir, 'sentence_embeddings.pickle'), 'wb') as f:
        pickle.dump(all_embs, f)
    with open(os.path.join(output_dir, 'sentence_meta.pickle'), 'wb') as f:
        pickle.dump(emb_meta, f)
# get dataset for DPOC
def get_dpoc(input_file_path, output_file_path):
    def get_words(sent):
        res = {k:[] for k in regex_dict}
        doc = nlp(sent)
        matches = matcher(doc)
        for match_id, start, end in matches:
            matched_span = doc[start:end]
            res[nlp.vocab.strings[match_id]].append(matched_span.text)

        return res
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(_)['conversations'] for _ in f]
    
    # inconsistency
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    regex_dict = {
        'VN': [{'POS': 'VERB'}, 
            {'POS': 'NOUN'}],
        'NOUN': [{'POS': 'NOUN'}],
        'NUM': [{'POS': 'NUM'}]
    }

    for k, v in regex_dict.items():
        matcher.add(k, [v])

    for i in range(len(data_list)):
        profile = data_list[i]['profile']
        pos = data_list[i]['pos']
        wrong_info = pos
        word1 = get_words(profile)
        word2 = get_words(pos)
        for k, v in word2.items():
            if len(v) == 0:
                continue
            
            profile_words = word1[k]
            for src_word in v:
                if src_word in profile_words:
                    profile_words.remove(src_word)
            if len(profile_words) == 0:
                continue
            for src_word in v:
                tgt_word = random.sample(profile_words, 1)[0]
                wrong_info = wrong_info.replace(src_word, tgt_word)
        if wrong_info == pos:
            wrong_info = ''
        data_list[i]['wrong_info'] = wrong_info
        if i % 1000 == 0:
            print(i)

    # fabrication
    with open('../dataset/ConvAI2/embedding/sentence_embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f)
    with open('../dataset/ConvAI2/embedding/sentence_meta.pickle', 'rb') as f:
        meta_data = pickle.load(f)
    embeddings = np.array(embeddings)
    res = faiss.StandardGpuResources()  # 使用标准GPU资源
    index = faiss.IndexFlatL2(768)
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embeddings)
    bot_idx = np.array([i for i in range(len(meta_data)) if meta_data[i]['role'] == 'bot'])
    bot_embs = embeddings[bot_idx]
    global_similarity_sents = []
    for i in range(0, bot_embs.shape[0], 10):
        inputs = bot_embs[i:i+10]
        D, I = index.search(inputs, 100)
        I = list(I)
        for _ in I:
            global_similarity_sents.append(_)
    global_similarity_sents = [list(_) for _ in global_similarity_sents]
    global_sent_dict = {}
    for idx, sents in zip(list(bot_idx), global_similarity_sents):
        src_sent = meta_data[idx]['text']
        tgt_sent = meta_data[sents[-1]]['text']
        global_sent_dict[src_sent] = tgt_sent
    for i in range(len(data_list)):
        pos = data_list[i]['pos']
        fictitious_info = global_sent_dict[pos]
        data_list[i]['fictitous_info'] = fictitious_info

    # inversion
    with open('../dataset/ConvAI2/embedding/sentence_embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f)
    with open('../dataset/ConvAI2/embedding/sentence_meta.pickle', 'rb') as f:
        meta_data = pickle.load(f)
    with open('../dataset/ConvAI2/embedding/profile_idx.json', 'r') as f:
        profile_idx = json.load(f)
    idx_embs = {}
    sent_emb = {}
    for i in range(len(meta_data)):
        idx = meta_data[i]['profile_idx']
        sent_emb[meta_data[i]['text']] = embeddings[i]
        if meta_data[i]['role'] != 'user':
            continue
        if idx not in idx_embs:
            idx_embs[idx] = [[], []]
        idx_embs[idx][0].append(meta_data[i]['text'])
        idx_embs[idx][1].append(embeddings[i])

    for k, v in idx_embs.items():
        idx_embs[k][1] = np.array(v[1])
    res = faiss.StandardGpuResources()  # 使用标准GPU资源
    index = faiss.IndexFlatL2(768)
    index = faiss.index_cpu_to_gpu(res, 0, index)
    cur_idx = -1
    for i in range(len(data_list)):
        profile = data_list[i]['profile']
        idx = profile_idx[profile]
        if idx != cur_idx:
            cur_idx = idx
            embs = idx_embs[idx][1]
            index.reset()
            index.add(embs)
        text = idx_embs[idx][0]
        pos = data_list[i]['pos']
        emb = sent_emb[pos]
        D, I = index.search(np.array([emb]), 1)
        I = list(I)[0]
        tgt_sent = text[I[0]]
        data_list[i]['identity_confusion'] = tgt_sent

    for i in range(len(data_list)):
        data = data_list[i]
        criterion = [data['wrong_info'], data['fictitous_info'], data['identity_confusion']]
        criterion = random.sample(criterion, 1)[0]
        data_list[i]['criterion'] = criterion
        with open(output_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'conversations': data}, ensure_ascii=False))
            f.write('\n')



if __name__ == '__main__':
    # get SFT dataset
    txt_path = './ConvAI2/train_self_original.txt'
    json_path = './ConvAI2/train_self_original.json'
    dataset_path = './ConvAI2/train_ConvAI2.json'
    dataset_msl_path = './ConvAI2/train_ConvAI2_msl.json'
    convert(txt_path, json_path, 'sft')
    extract(json_path, dataset_path, 'sft')
    get_msl(dataset_path, dataset_msl_path)

    # get DPO dataset
    txt_path = './ConvAI2/train_self_original_cands.txt'
    json_path = './ConvAI2/train_self_original_cands.json'
    dataset_path = './ConvAI2/train_ConvAI2_cands.json'
    dataset_dpo_path = './ConvAI2/train_ConvAI2_dpo.json'
    convert(txt_path, json_path, 'sft')
    extract(json_path, dataset_path, 'sft')
    get_dpo(dataset_path, dataset_dpo_path)

    # get dpoc dataset
    dataset_dpo_path = './ConvAI2/train_ConvAI2_dpo.json'
    dataset_dpoc_path = './ConvAI2/train_ConvAI2_dpoc.json'
    get_dpoc(dataset_dpo_path, dataset_dpoc_path)