import json
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import numpy as np
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import InputExample
import numpy as np

#### coherence & diversity ####
def diversity_score(text, n):
    tokens = word_tokenize(text)
    ngram_counts = nltk.ngrams(tokens, n)
    unique_ngrams = set(ngram_counts)
    ngram_counts = list(nltk.ngrams(tokens, n))
    if len(ngram_counts) == 0:
        return 0
    diversity = len(unique_ngrams) / len(ngram_counts)
    return diversity

def compute_metric(src):
    with open(src, 'r', encoding='utf-8') as f:
        data_list = [(json.loads(_)['llama_response'], json.loads(_)['response']) for _ in f]
    bleu = 0
    rouge_1, rouge_2, rouge_l = 0, 0, 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    distinct_1, distinct_2 = 0, 0
    for src, tgt in data_list:
        bleu_score = sentence_bleu([src], tgt, weights=(1, 0, 0, 0))
        bleu += bleu_score

        rouge_scores = scorer.score(src, tgt)
        rouge_1 += rouge_scores['rouge1'].fmeasure
        rouge_2 += rouge_scores['rouge2'].fmeasure
        rouge_l += rouge_scores['rougeL'].fmeasure

        distinct_1 += diversity_score(src, 1)
        distinct_2 += diversity_score(src, 2)

    print('bleu: {}\nrouge-1: {}\nrouge-2: {}\nrouge-l: {}\ndistinct-1: {}\ndistinct-2: {}'.format(bleu/len(data_list), rouge_1/len(data_list), rouge_2/len(data_list), rouge_l/len(data_list), distinct_1/len(data_list), distinct_2/len(data_list)))

#### p_score ####
def cal_s_for_each_history(r, h, idf_dict):
	c = 0
	has_c = {}
	for w in r:
		if w in h and w not in has_c:
			c += idf_dict[w]
			has_c[w] = 1
	return c

def docs(w, history_list):
	c = 0
	for i,h in enumerate(history_list):
		if w in h:
			c += 1
	return c

def gen_idf_dict(history_list):
	idf_dict = {}
	for i, h in enumerate(history_list):
		for w in h:
			if w not in idf_dict:
				idf = math.log(len(history_list) *1.0 / docs(w, history_list))
				idf_dict[w] = idf

	return idf_dict

def cal_p_f1(history, response):
	history = ' '.join(history)
	history_words = word_tokenize(history)
	response_words = word_tokenize(response)
	set1 = set(history_words)
	set2 = set(response_words)

	true_positives = len(set1.intersection(set2))
	false_positives = len(set1 - set2)
	false_negatives = len(set2 - set1)

	precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
	recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

	f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
	return f1_score
	

def cal_p_cover(history, response):
	s_sum = 0
	idf_dict = gen_idf_dict(history)
	a1 = sorted(idf_dict.items(),key = lambda x:x[1],reverse = True)
	# print(a1)
	s_list = []
	for i, h in enumerate(history):
		h = ' '.join(h).replace("<PAD>","").replace("<EOS>","").replace("<SOS>","").split()
		r = ' '.join(response).replace("<EOS>","").split()
		s = cal_s_for_each_history(r, h, idf_dict)
		s_list.append(s)
	s_max = max(s_list)
	s_sum += s_max
	return (s_sum+0.0)

def compute_p_score(src):
    with open(src, 'r', encoding='utf-8') as f:
        data_list = [(json.loads(_)['history'][1::2][-1:], json.loads(_)['llama_response']) for _ in f]

    p_f1 = 0
    p_cover = 0
    for history, response in data_list:
        p_f1 += cal_p_f1(history, response)
        p_cover += cal_p_cover(history, response)

    print('p_f1: {}\np_cover: {}'.format(p_f1/len(data_list), p_cover/len(data_list)))


#### persona ####
def get_dataloader(input_examples, tokenizer, device, batch_size=256):
    features = convert_examples_to_features(
        input_examples,
        tokenizer,
        label_list=['0', '1'],
        max_length=128,
        output_mode='classification',
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
    dataset = TensorDataset(all_input_ids, all_attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def read_data(file_path):
    with open(file_path, 'r') as f:
        data_list = [json.loads(_) for _ in f]
    personas = [_['profile'].split('\n') for _ in data_list]
    preds = [_['llama_response'] for _ in data_list]

    examples = []
    cnt = 0
    for persona_list, hyp in zip(personas, preds):
        for persona in persona_list:
            examples.append(InputExample(str(cnt), persona, hyp, '0'))
            cnt += 1
    
    return examples, len(preds)

def cal_coh_score(scores):
    t = 0
    for score in scores:
        if score > 0:
            t += 1
    return t / len(scores)

def cal_coh_con_score(scores):
    t = 0
    for score in scores:
        if score == 2:
            t += 1
    return t / len(scores)

def compute_persona(src, model, tokenizer, device):
    input_examples, num = read_data(src)
    train_dataloader = get_dataloader(input_examples, tokenizer, device, batch_size=512)
    all_logits = None
    with torch.no_grad():
        for batch in train_dataloader:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = model(**inputs)
            if all_logits is None:
                all_logits = outputs[0].cpu().detach()
            else:
                all_logits = torch.cat((all_logits, outputs[0].cpu().detach()), dim=0)

    results = torch.argmax(all_logits, dim=1)
    scores = list(results.numpy())
    print(cal_coh_score(scores))
    print(cal_coh_con_score(scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', default='coherence', choices=['coherence', 'p_score', 'persona'])
    parser.add_argument('--src', default='results/ConvAI2/valid_ConvAI2')
    parser.add_argument('--NLI_model', default='models/NLI')
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    if args.evaluation == 'coherence':
        compute_metric(args.src)
    elif args.evaluation == 'p_score':
        compute_p_score(args.src)
    elif args.evaluation == 'persona':
        tokenizer = AutoTokenizer.from_pretrained(args.NLI_model)
        model = AutoModelForSequenceClassification.from_pretrained(args.NLI_model)
        device = torch.device(args.device)
        model.to(device)
        model.eval()
        compute_persona(args.src, model, tokenizer, args.device)