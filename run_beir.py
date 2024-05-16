import logging
import torch
import re
import os
from beir import util
from enum import Enum
from tqdm import tqdm
import pytrec_eval
import argparse

from beir.datasets.data_loader import GenericDataLoader
from xtr.retrieval_xtr import XtrRetriever

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="google/xtr-base-en")
parser.add_argument("--dataset", type=str, default="scifact")
parser.add_argument("--use_faiss", action="store_true")
parser.add_argument("--doc_sample_ratio", type=float, default=0.2)
parser.add_argument("--vec_sample_ratio", type=float, default=0.2)
parser.add_argument("--code_size", type=int, default=64)
parser.add_argument("--nprobe", type=int, default=128)
parser.add_argument("--token_top_k", type=int, default=8000)
parser.add_argument("--dataset_dir", type=str, default="datasets")
parser.add_argument("--index_dir", type=str, default="index")
parser.add_argument("--load_index", action="store_true")

args = parser.parse_args()

######################################
print("Step 1 - Load XTR Retriever")
######################################

xtr = XtrRetriever(
    model_name_or_path=args.model_name_or_path,
    use_faiss=args.use_faiss, 
    device=device
)

######################################
print("Step 2 - Load BEIR Datasets")
######################################

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
logger.info("Dataset downloaded here: {}".format(data_path))

data_path = f"{args.dataset_dir}/{args.dataset}"

if args.dataset == 'msmarco':
    split='dev'
else:
    split='test'
corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

######################################
print("Step 3 - Index BEIR Corpus")
######################################

# For Scifact + XTR-base-en (P100), this should take about 3 minutes.
all_docs = []
all_keys = []
for doc_key, doc in tqdm(corpus.items()):
    doc_text = f"{doc['title']} {doc['text']}".lower()
    all_keys.append(doc_key)
    all_docs.append(doc_text)

index_dir = f"{args.index_dir}/{args.dataset}"
if args.load_index:
    index_num = xtr.load_index(
        all_docs, 
        index_dir=index_dir, 
        code_size=args.code_size, 
        nprobe=args.nprobe
    )
else:
    index_num = xtr.build_index(
        all_docs, 
        index_dir=index_dir, 
        doc_sample_ratio=args.doc_sample_ratio,
        vec_sample_ratio=args.vec_sample_ratio,
        code_size=args.code_size, 
        nprobe=args.nprobe
    )
print(f"XTR Index Size: {index_num}")

######################################
print("Step 4 - Run BEIR Evaluation")
######################################

# For Scifact, XTR-base-en (P100), this should take about 2 minutes.

# Evaluation hyperparameters.
TOKEN_TOP_K = args.token_top_k
TREC_TOP_K = 100

predictions = {}
# Running evaluation per query for a better latency measurement.
for q_idx, (query_key, query) in tqdm(enumerate(queries.items()), total=len(queries)):
    ranking, metadata = xtr.retrieve_docs(
        [query.lower()],
        token_top_k=TOKEN_TOP_K,
        return_text=False
    )
    ranking = ranking[0]
    predictions[query_key] = {str(all_keys[did]): score for did, score in ranking[:TREC_TOP_K]}

# Run pytrec_eval.
K_VALUES = [5, 10, 50, 100]
METRIC_NAMES = ['ndcg_cut', 'map_cut', 'recall']

def eval_metrics(qrels, predictions):
    measurements = []
    for metric_name in METRIC_NAMES:
        measurements.append(
            f"{metric_name}." + ",".join([str(k) for k in K_VALUES])
        )
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measurements)
    final_scores = evaluator.evaluate(predictions)

    final_metrics = dict()
    for metric_name in METRIC_NAMES:
        for k in K_VALUES:
            final_metrics[f"{metric_name}@{k}"] = 0.0

    for query_id in final_scores.keys():
        for metric_name in METRIC_NAMES:
            for k in K_VALUES:
                final_metrics[f"{metric_name}@{k}"] += final_scores[query_id][
                    f"{metric_name}_{k}"
                ]

    for metric_name in METRIC_NAMES:
        for k in K_VALUES:
            final_metrics[f"{metric_name}@{k}"] = round(
                final_metrics[f"{metric_name}@{k}"] / len(final_scores), 5
            )

    print("[Result]")
    for metric_name, metric_score in final_metrics.items():
        metric_score = round(metric_score*100,2)
        print(f"{metric_name}: {metric_score}")

eval_metrics(qrels, predictions)