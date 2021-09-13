import os
import time
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataset import SelectionDataset
from transform import SelectionSequentialTransform, SelectionJoinTransform, SelectionConcatTransform
from encoder import PolyEncoder, BiEncoder, CrossEncoder

from sklearn.metrics import label_ranking_average_precision_score

import logging
logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--bert_model", default='/search/odin/guobk/data/data_polyEncode/vpa/model_small_all', type=str)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--model_type", default='bert', type=str)
parser.add_argument("--output_dir", default='/search/odin/guobk/data/data_polyEncode/vpa/model_small_all', type=str)
parser.add_argument("--train_dir", default='/search/odin/guobk/data/data_polyEncode/vpa/train_data_all/', type=str)

parser.add_argument("--use_pretrain", action="store_true")
parser.add_argument("--architecture", default='poly', type=str, help='[poly, bi, cross]')

parser.add_argument("--max_contexts_length", default=32, type=int)
parser.add_argument("--max_response_length", default=64, type=int)
parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
parser.add_argument("--print_freq", default=100, type=int, help="Log frequency")

parser.add_argument("--poly_m", default=16, type=int, help="Number of m of polyencoder")

parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--warmup_steps", default=100, type=float)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

parser.add_argument("--num_train_epochs", default=10.0, type=float,
                                        help="Total number of training epochs to perform.")
parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
print(args)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
set_seed(args)
def eval_running_model(dataloader, test=False):
    model.eval()
    eval_loss, eval_hit_times = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    r10 = r2 = r1 = r5 = 0
    mrr = []
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        if args.architecture == 'cross':
            text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch = batch
            with torch.no_grad():
                logits = model(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch)
                loss = F.cross_entropy(logits, torch.argmax(labels_batch, 1))
        else:
            context_token_ids_list_batch, context_input_masks_list_batch, \
            response_token_ids_list_batch, response_input_masks_list_batch, labels_batch = batch
            with torch.no_grad():
                logits = model(context_token_ids_list_batch, context_input_masks_list_batch,
                                              response_token_ids_list_batch, response_input_masks_list_batch)
                loss = F.cross_entropy(logits, torch.argmax(labels_batch, 1))
        r2_indices = torch.topk(logits, 2)[1] # R 2 @ 100
        r5_indices = torch.topk(logits, 5)[1] # R 5 @ 100
        r10_indices = torch.topk(logits, 10)[1] # R 10 @ 100
        r1 += (logits.argmax(-1) == 0).sum().item()
        r2 += ((r2_indices==0).sum(-1)).sum().item()
        r5 += ((r5_indices==0).sum(-1)).sum().item()
        r10 += ((r10_indices==0).sum(-1)).sum().item()
        # mrr
        logits = logits.data.cpu().numpy()
        for logit in logits:
            y_true = np.zeros(len(logit))
            y_true[0] = 1
            mrr.append(label_ranking_average_precision_score([y_true], [logit]))
        eval_loss += loss.item()
        nb_eval_examples += labels_batch.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = r1 / nb_eval_examples
    if not test:
        result = {
            'train_loss': tr_loss / nb_tr_steps,
            'eval_loss': eval_loss,
            'R1': r1 / nb_eval_examples,
            'R2': r2 / nb_eval_examples,
            'R5': r5 / nb_eval_examples,
            'R10': r10 / nb_eval_examples,
            'MRR': np.mean(mrr),
            'epoch': epoch,
            'global_step': global_step,
        }
    else:
        result = {
            'eval_loss': eval_loss,
            'R1': r1 / nb_eval_examples,
            'R2': r2 / nb_eval_examples,
            'R5': r5 / nb_eval_examples,
            'R10': r10 / nb_eval_examples,
            'MRR': np.mean(mrr),
        }

    return result
class Tokenizer(object):
    def __init__(self,path_vocab,do_lower_case):
        with open(path_vocab,'r') as f:
            self.vocab = f.read().strip().split('\n')
        self.vocab = {self.vocab[k]:k for k in range(len(self.vocab))}
        self.do_lower_case = do_lower_case
    def token_to_ids(self,text,max_len,is_context=True):
        if type(text)==str:
            text = text.strip()
            if self.do_lower_case:
                text = text.lower()
            res = [self.vocab['[CLS]']]
            for i in range(min(max_len-2,len(text))):
                if text[i] not in self.vocab:
                    res.append(self.vocab['[MASK]'])
                else:
                    res.append(self.vocab[text[i]])
            res.append(self.vocab['[SEP]'])
            segIds = []
            segIds = [1 for _ in range(len(res))]
            segIds = segIds+[0]*(max_len-len(segIds))
            res = res[:max_len]
            res = res + [0]*(max_len-len(res))
            tokenIds = res
            return tokenIds,segIds
        else:
            tokenIds,segIds = [], []
            for t in text:
                res = self.token_to_ids(t, max_len)
                tokenIds.append(res[0])
                segIds.append(res[1])
        return tokenIds,segIds
def testLoader(queries, Docs, mytokenzier):
    Context_ids, Context_msk = mytokenizer.token_to_ids(queries,max_len=args.max_contexts_length)
    Response_ids, Response_msk = mytokenizer.token_to_ids(Docs,max_len=args.max_response_length)


if __name__ == '__main__':

    MODEL_CLASSES = {
        'bert': (BertConfig, BertTokenizerFast, BertModel),
    }
    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]

    ## init dataset and bert model
    tokenizer = TokenizerClass.from_pretrained(os.path.join(args.bert_model, "vocab.txt"), do_lower_case=True, clean_text=False)
    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=args.max_contexts_length)
    response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length)
    concat_transform = SelectionConcatTransform(tokenizer=tokenizer, max_len=args.max_response_length+args.max_contexts_length)

    print('=' * 80)
    print('Train dir:', args.train_dir)
    print('Output dir:', args.output_dir)
    print('=' * 80)

    state_save_path = os.path.join(args.output_dir, '{}_{}_pytorch_model.bin'.format(args.architecture, args.poly_m))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################
    ## build BERT encoder
    ########################################
    bert_config = ConfigClass.from_json_file(os.path.join(args.bert_model, 'config.json'))
    bert = BertModelClass(bert_config)

    if args.architecture == 'poly':
        model = PolyEncoder(bert_config, bert=bert, poly_m=args.poly_m)
    elif args.architecture == 'bi':
        model = BiEncoder(bert_config, bert=bert)
    elif args.architecture == 'cross':
        model = CrossEncoder(bert_config, bert=bert)
    else:
        raise Exception('Unknown architecture.')
    model.resize_token_embeddings(len(tokenizer)) 
    model.to(device)

    print('Loading parameters from', state_save_path)
    model.load_state_dict(torch.load(state_save_path))

    mytokenizer = Tokenizer(path_vocab=os.path.join(args.bert_model, "vocab.txt"),do_lower_case = True)

    with open('/search/odin/guobk/data/vpaSupData/Docs-0809.json','r') as f:
        Docs = json.load(f)
    with open('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json','r') as f:
        Queries = json.load(f)
    Docs = [d['content'] for d in Docs]
    Queries = [d['input'] for d in Queries]
    Context_ids, Context_msk = mytokenizer.token_to_ids(Queries,max_len=args.max_contexts_length)
    Response_ids, Response_msk = mytokenizer.token_to_ids(Docs,max_len=args.max_response_length)
    Response_ids, Response_msk = np.array(Response_ids), np.array(Response_msk)
    Response_ids, Response_msk = np.expand_dims(Response_ids,axis=0), np.expand_dims(Response_msk,axis=0)
    Response_ids, Response_msk = torch.from_numpy(Response_ids).to(device), torch.from_numpy(Response_msk).to(device)
    batch_size = 100
    Res = []
    for step in range(16, len(Context_ids)):
        context_ids = Context_ids[step]
        context_msk = Context_msk[step]
        context_ids, context_msk = np.array(context_ids), np.array(context_msk)
        context_ids, context_msk = np.expand_dims(context_ids,axis=0), np.expand_dims(context_msk,axis=0)
        context_ids, context_msk = torch.from_numpy(context_ids).to(device), torch.from_numpy(context_msk).to(device)
        ii = 0
        r = []
        while ii<Response_ids.shape[1]:
            response_ids = Response_ids[:,ii:(ii+batch_size),:]
            response_msk = Response_msk[:,ii:(ii+batch_size),:]
            sims = model(context_ids,context_msk, response_ids, response_msk)
            ii+=batch_size
            r.extend(list(sims.cpu().detach().numpy()[0]))
            print(step,ii)
        R = [[Docs[i],r[i]] for i in range(len(r))]
        R = sorted(R,key=lambda x:-x[-1])
        R = [d[0]+'\t'+str(d[1]) for d in R[:10]]
        Res.append({'input':Queries[step],'rec_poly':R})
        with open('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-poly.json','w') as f:
            json.dump(Res,f,ensure_ascii=False,indent=4)



        