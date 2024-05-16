# XTR-Pytorch

This repository provides a PyTorch-based reimplementation of XTR (Contextualized Token Retriever) for document retrieval. For this reimplemetation, I referred to the [original XTR code](https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb) by the author, which uses Tensorflow.

## Installation
```bash
$ git clone git@github.com:mjeensung/xtr-pytorch.git
$ pip install -e .
```

## Usage

### Simple example
To see how this XTR-pytorch works, please see the example snippet below or run `python run_sample.py`.

```python
# Create the dataset
sample_doc = "Google LLC (/ˈɡuːɡəl/ (listen)) is an American multinational technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, artificial intelligence..."
chunks = [chunk.lower() for chunk in sent_tokenize(sample_doc)]

# Load the XTR retriever
xtr = XtrRetriever(model_name_or_path="google/xtr-base-en", use_faiss=False, device="cuda")

# Build the index
xtr.build_index(chunks)

# Retrieve top-3 documents given the query
query = "Who founded google"
retrieved_docs, metadata = xtr.retrieve_docs([query], document_top_k=3)
for rank, (did, score, doc) in enumerate(retrieved_docs[0]):
    print(f"[{rank}] doc={did} ({score:.3f}): {doc}")

"""
>> [0] doc=0 (0.925): google llc (/ˈɡuːɡəl/ (listen)) is an american multinational technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, artificial intelligence, and consumer electronics.
>> [1] doc=1 (0.903): it has been referred to as "the most powerful company in the world" and one of the world's most valuable brands due to its market dominance, data collection, and technological advantages in the area of artificial intelligence.
>> [2] doc=2 (0.900): its parent company alphabet is considered one of the big five american information technology companies, alongside amazon, apple, meta, and microsoft.
"""
```

### BEIR example

To evaluate XTR-pytorch on the [BEIR benchmark](https://github.com/beir-cellar/beir/), please run `run_beir.py`.
```bash
$ pip install beir --no-deps
$ python run_beir.py \
    --model_name_or_path google/xtr-base-en \
    --dataset nfcorpus \
    --token_top_k 8000 \
    --use_faiss
```

Below is the comparsion of NDCG@10 between the reported scores from [the XTR paper](https://arxiv.org/abs/2304.01982) and the scores from the reimplemented XTR in this repo across four datasets from BEIR.

|              Dataset                | XTR base ([Reported](https://arxiv.org/abs/2304.01982), k=40000) | XTR-pytorch base (This repo, k=8000) |
|:----------------------------------|:--------:|:--------:|
| MSMARCO | 45.0 | 42.9 |
| NQ | 53.0 | 52.0 |
| NFCorpus | 34.0 | 34.1 |
| SciFact | 71.0 | 71.8 |

