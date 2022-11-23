import pandas as pd
import torch
import os
from tqdm import tqdm, trange

from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, AdamW

import argparse



def get_special_tokens(tokenizer):
    pad_tok = tokenizer.vocab["[PAD]"]
    sep_tok = tokenizer.vocab["[SEP]"]
    cls_tok = tokenizer.vocab["[CLS]"]

    return pad_tok, sep_tok, cls_tok


def distinctness_all(generations_df):
    unigrams, bigrams, trigrams = [], [], []

    # calculate dist1, dist2, dist3 across generations for every prompt
    for i in trange(len(generations_df), desc='Evaluating diversity'):
        generations = str(generations_df[0][i])
        o = generations.split(' ')
        unigrams += o
        for i in range(len(o) - 1):
            bigrams.append(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.append(o[i] + '_' + o[i + 1] + '_' + o[i + 2])

    dist1 = len(set(unigrams)) * 1.0 / len(unigrams)
    dist2 = len(set(bigrams)) * 1.0 / len(bigrams)
    dist3 = len(set(trigrams)) * 1.0 / len(trigrams)
    # take the mean across prompts
    return dist1, dist2, dist3


def init_config():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--dataset', type=str,default=" ", help='dataset to use')
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--att_gamma', type=float, default=0.0)
    parser.add_argument('--decoding_strategy', type=str, choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument('--kl_att', type=float, default=1.0, help="attention KL weight")
    parser.add_argument('--adv_weight', type=float, default=1.0, help="adv weight")
    parser.add_argument('--file_name', type=str, default=None, help='generated text file name')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


args = init_config()

bert_file = "./clf/bert_clf"

torch.cuda.set_device(args.gpu)


tokenizer = BertTokenizer.from_pretrained(bert_file, do_lower_case=True)
pad_tok, sep_tok, cls_tok = get_special_tokens(tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu: ", n_gpu)
print(torch.cuda.is_available())

model = BertForSequenceClassification.from_pretrained(bert_file)


model.cuda()


file_path = args.file_name

report_name = file_path[:-4] + "_result.txt"
test = pd.read_csv(file_path, header=None)
dist1, dist2, dist3 = distinctness_all(test)
for i, dist_n in enumerate([dist1, dist2, dist3]):
    print(f'dist-{i + 1} = {dist_n}\n')


test_sentence = test[0]
predict_labels = list(test[1])

loss_fct_mse = torch.nn.MSELoss()
loss_fct = torch.nn.L1Loss()

labels = []
eval_loss = 0

for i in tqdm(range(len(test_sentence))):
    inputs = tokenizer(str(test_sentence[i]), return_tensors="pt", truncation=True, max_length=512)
    model.eval()
    inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    logits = torch.sigmoid(outputs.logits).item()
    labels.append(logits)

eval_loss = loss_fct(torch.FloatTensor(labels), torch.FloatTensor(predict_labels))
eval_loss_mse = loss_fct_mse(torch.FloatTensor(labels), torch.FloatTensor(predict_labels))


print(f"MAE loss:\n {eval_loss}")
print(f"MSE loss:\n {eval_loss_mse}")

with open(report_name, 'w', newline='\n') as ott:
    for i, dist_n in enumerate([dist1, dist2, dist3]):
        ott.write(f'dist-{i + 1} = {dist_n}\n')
    ott.write(f"MAE loss:\n {eval_loss}")
    ott.write(f"MSE loss:\n {eval_loss_mse}")





