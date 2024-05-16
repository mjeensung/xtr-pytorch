# coding=utf-8
# Copyright 2024, Jinhyuk Lee and Mujeen Sung. All rights reserved.
# Original repo for XTR: https://github.com/google-deepmind/xtr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XTR Retriever model implementation."""


from tqdm import tqdm
import numpy as np
import logging
import faiss
import torch
import os
import math
import random

from faiss import write_index, read_index
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoTokenizer
from transformers.utils import logging
from .modeling_xtr import XtrModel
from .configuration_xtr import XtrConfig

logger = logging.get_logger(__name__)

class XtrRetriever(object):
    def __init__(self, 
                model_name_or_path: str,
                cache_dir: Optional[str] = None,
                use_faiss=False, 
                device='cpu'
        ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.config = XtrConfig(model_name_or_path)
        self.encoder = XtrModel(model_name_or_path, config=self.config, device=device)
        self.use_faiss = use_faiss
        self.device = device
        self.max_seq_len = self.tokenizer.model_max_length
        self.token_embed_dim = self.encoder.get_token_embed_dim()
        self.doc_offset = 512 # max token length

    def tokenize(self, text):
        return [self.tokenizer.id_to_string(id_).numpy().decode('utf-8') for id_ in self.tokenizer.tokenize(text)]

    def get_token_embeddings(self, texts):
        encoded_inputs = self.tokenizer(texts, return_tensors='pt',padding='max_length', truncation=True, max_length=self.max_seq_len).to(device=self.device)

        with torch.no_grad():
            batch_embeds = self.encoder(**encoded_inputs)
            
        batch_lengths = np.sum(encoded_inputs['attention_mask'].cpu().numpy(), axis=1)
        
        return batch_embeds.cpu().numpy(), batch_lengths

    def get_flatten_embeddings(self, batch_text, return_last_offset=False):
        batch_embeddings, batch_lengths = self.get_token_embeddings(batch_text)
        flatten_embeddings = None
        num_tokens = 0
        offsets = [0]
        for batch_id, (embeddings, length) in enumerate(zip(batch_embeddings, batch_lengths)):
            if flatten_embeddings is not None:
                flatten_embeddings = np.append(flatten_embeddings, embeddings[:int(length)], axis=0)
            else:
                flatten_embeddings = embeddings[:int(length)]
            num_tokens += int(length)
            offsets.append(num_tokens)
        assert num_tokens == flatten_embeddings.shape[0]
        if not return_last_offset:
            offsets = offsets[:-1]
        return flatten_embeddings, offsets
    
    def build_index(self, documents, batch_size=32, **kwargs):
        if self.use_faiss:
            index_num = self.build_index_faiss(documents, batch_size=batch_size, **kwargs)
        else:
            index_num = self.build_index_bruteforce(documents, batch_size=batch_size)

        return index_num

    def build_index_bruteforce(self, documents, index_dir=None, batch_size=32):
        # Used only for small-scale, exact inference.
        all_token_embeds = np.zeros((len(documents)*self.max_seq_len, self.token_embed_dim), dtype=np.float32)
        num_tokens = 0
        for batch_idx in tqdm(range(0, len(documents), batch_size)):
            batch_docs = documents[batch_idx:batch_idx+batch_size]
            batch_embeds, batch_offsets = self.get_flatten_embeddings(batch_docs)
            num_tokens += len(batch_embeds)
            all_token_embeds[num_tokens-len(batch_embeds):num_tokens] = batch_embeds

        class BruteForceSearcher(object):
            def search_batched(self, query_embeds, final_num_neighbors, **kwargs):
                scores = query_embeds.dot(all_token_embeds[:num_tokens].T) # Q x D
                top_ids = scores.argsort(axis=1)[:, ::-1][:,:final_num_neighbors] # Q x top_k
                return top_ids, [q_score[q_top_ids] for q_score, q_top_ids in zip(scores, top_ids)] # (Q x top_k, Q x top_k)
        self.searcher = BruteForceSearcher()
        self.docs = documents
        print("Index Ready!", self.searcher)

        return num_tokens

    def build_index_faiss(self, documents, batch_size=32, doc_sample_ratio=0.2, vec_sample_ratio=0.2, seed=29, index_dir=None, code_size=64, nprobe=4):
        # 1. sample token embeddings for train index
        random.seed(seed)
        np.random.seed(seed)
        smpl_vec_len = int(vec_sample_ratio * self.max_seq_len)
        smpl_documents = random.sample(documents, int(doc_sample_ratio * len(documents)))
        smpl_token_embeds = np.zeros((int(len(documents)*doc_sample_ratio*self.max_seq_len*vec_sample_ratio), self.token_embed_dim), dtype=np.float32)
        num_tokens = 0
        for batch_idx in tqdm(range(0, len(smpl_documents), batch_size)):
            batch_docs = smpl_documents[batch_idx:batch_idx+batch_size]
            batch_embeds, _ = self.get_flatten_embeddings(batch_docs)
            smpl_batch_idx = np.random.choice(len(batch_embeds), int(vec_sample_ratio * len(batch_embeds)))
            smpl_batch_embeds = batch_embeds[smpl_batch_idx]
            num_tokens += len(smpl_batch_embeds)
            smpl_token_embeds[num_tokens-len(smpl_batch_embeds):num_tokens] = smpl_batch_embeds

        smpl_token_embeds = smpl_token_embeds[:num_tokens]

        # use the square root of total token nums as num_clusters
        num_clusters = int(math.sqrt(num_tokens/doc_sample_ratio/vec_sample_ratio))

        ds = self.token_embed_dim
        quantizer = faiss.IndexFlatIP(ds)
        opq_matrix = faiss.OPQMatrix(ds, code_size)
        opq_matrix.niter = 10
        sub_index = faiss.IndexIVFPQ(quantizer, ds, num_clusters, code_size, 8, faiss.METRIC_INNER_PRODUCT)
        sub_index.nprobe = nprobe
        index = faiss.IndexPreTransform(opq_matrix, sub_index)
        # Convert to GPU index
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
        gpu_index.verbose = False

        # Train on GPU with sampled token embeddings and back to CPU
        gpu_index.train(smpl_token_embeds)
        index = faiss.index_gpu_to_cpu(gpu_index)

        # 2. embed all tokens in batch and add them faiss index
        add_size = 128
        add_num_tokens = 0
        add_count = 0
        add_token_ids = []
        add_token_embeds = np.zeros((int(batch_size*add_size*self.max_seq_len), self.token_embed_dim), dtype=np.float32)
        all_num_tokens = 0
        for batch_idx in tqdm(range(0, len(documents), batch_size)):
            batch_docs = documents[batch_idx:batch_idx+batch_size]
            batch_embeds, batch_offsets = self.get_flatten_embeddings(batch_docs)
            batch_token_len = [batch_offsets[i+1] - batch_offsets[i] for i, offset in enumerate(batch_offsets[:-1])] + [len(batch_embeds) - batch_offsets[-1]]
            # batch_token_ids = [f"{did*self.doc_offset + tid}" for did in range(batch_idx,batch_idx+len(batch_docs)) for tid in range(batch_token_len[did - batch_idx])]
            batch_token_ids = [did*self.doc_offset + tid for did in range(batch_idx,batch_idx+len(batch_docs)) for tid in range(batch_token_len[did - batch_idx])]

            add_num_tokens += len(batch_embeds)
            all_num_tokens += len(batch_embeds)
            add_token_embeds[add_num_tokens-len(batch_embeds):add_num_tokens] = batch_embeds
            add_token_ids += batch_token_ids

            # add batch embeds with ids to index 
            add_count += 1
            if add_count >= add_size:
                add_token_embeds = add_token_embeds[:len(add_token_ids)]
                index.add_with_ids(x=add_token_embeds,ids=np.array(add_token_ids))
                
                add_num_tokens = 0
                add_count = 0
                add_token_ids = []
                add_token_embeds = np.zeros((int(batch_size*add_size*self.max_seq_len), self.token_embed_dim), dtype=np.float32)
        
        if add_count != 0:
            add_token_embeds = add_token_embeds[:len(add_token_ids)]
            index.add_with_ids(x=add_token_embeds,ids=np.array(add_token_ids))
        
        assert all_num_tokens == index.ntotal

        self.save_index(index, index_dir, code_size, nprobe)

        class FaissSearcher(object):
            def search_batched(self, query_embeds, final_num_neighbors, **kwargs):
                scores, top_ids = index.search(query_embeds, final_num_neighbors)
                return top_ids, scores
        self.searcher = FaissSearcher()
        self.docs = documents

        print("Index Ready!", self.searcher)
        return index.ntotal


    def save_index(self, index, index_dir, code_size, nprobe):
        os.makedirs(index_dir, exist_ok=True)
        index_path = f"{index_dir}/cs{code_size}.index"
        write_index(index, index_path)

    def load_index(self, documents, index_dir, code_size, nprobe):
        self.docs = documents
        
        index_path = f"{index_dir}/cs{code_size}.index"
        
        index = read_index(index_path)
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = nprobe
        
        class FaissSearcher(object):
            def search_batched(self, query_embeds, final_num_neighbors, **kwargs):
                scores, top_ids = index.search(query_embeds, final_num_neighbors)
                return top_ids, scores
        self.searcher = FaissSearcher()

        return index.ntotal
        
    def batch_search_tokens(self, batch_query, token_top_k=100):
        all_query_encodings, query_offsets = self.get_flatten_embeddings(batch_query, return_last_offset=True)
        all_neighbors, all_scores = self.searcher.search_batched(all_query_encodings, final_num_neighbors=token_top_k)

        return [
            (
                [f'q_{i}' for i in range(query_offsets[oid], query_offsets[oid+1])],  # query_id
                all_neighbors[query_offsets[oid]:query_offsets[oid+1]],  # neighbors
                all_scores[query_offsets[oid]:query_offsets[oid+1]],  # scores
            )
            for oid in range(len(query_offsets)-1)
        ]

    def estimate_missing_similarity(self, batch_result):
        batch_qtoken_to_ems = [dict() for _ in range(len(batch_result))]
        for b_idx, (query_tokens, _, distances) in enumerate(batch_result):
            for token_idx, qtoken in enumerate(query_tokens):
                idx_t = (token_idx, qtoken)
                # Use similarity of the last token as imputed similarity.
                batch_qtoken_to_ems[b_idx][idx_t] = distances[token_idx][-1]
        return batch_qtoken_to_ems

    def aggregate_scores(self, batch_result, batch_ems, document_top_k):
        """Aggregates token-level retrieval scores into query-document scores."""
        
        def get_did2scores(query_tokens, all_neighbors, all_scores):
            did2scores = {}
            # |Q| x k'
            for qtoken_idx, (qtoken, neighbors, scores) in enumerate(zip(query_tokens, all_neighbors, all_scores)):
                for _, (doc_token_id, score) in enumerate(zip(neighbors, scores)):
                    if np.isnan(score):
                        continue

                    docid = doc_token_id // self.doc_offset
                    if docid not in did2scores:
                        did2scores[docid] = {}
                    qtoken_with_idx = (qtoken_idx, qtoken)
                    if qtoken_with_idx not in did2scores[docid]:
                        # Only keep the top score for sum-of-max.
                        did2scores[docid][qtoken_with_idx] = score
            
            return did2scores
        batch_did2scores = [get_did2scores(qtokens, neighbors, scores) for qtokens, neighbors, scores in batch_result]

        def add_ems(did2scores, query_tokens, ems):
            # |Q| x |Q|k' (assuming most docid is unique)
            for qtoken_idx, qtoken in enumerate(query_tokens):
                qtoken_with_idx = (qtoken_idx, qtoken)
                for docid, scores in did2scores.items():
                    if qtoken_with_idx not in scores:
                        scores[qtoken_with_idx] = ems[qtoken_with_idx]
        for did2scores, result, ems in zip(batch_did2scores, batch_result, batch_ems):
            add_ems(did2scores, result[0], ems)

        def get_final_score(did2scores, query_tokens):
            final_qd_score = {}
            # |Q|k' x |Q|
            for docid, scores in did2scores.items():
                assert len(scores) == len(query_tokens)
                final_qd_score[docid] = sum(scores.values()) / len(scores)
            return final_qd_score

        batch_scores = [get_final_score(did2scores, result[0]) for did2scores, result in zip(batch_did2scores, batch_result)]

        batch_ranking = [
            sorted([(docid, score) for docid, score in final_qd_score.items()], key=lambda x: x[1], reverse=True)[:document_top_k]
            for final_qd_score in batch_scores
        ]
        return batch_ranking
    
    def get_document_text(self, batch_ranking):
        batch_retrieved_docs = []
        for ranking in batch_ranking:
            retrieved_docs = []
            for did, score in ranking:
                retrieved_docs.append((did, score, self.docs[did]))
            batch_retrieved_docs.append(retrieved_docs)
        return batch_retrieved_docs

    def retrieve_docs(
        self,
        batch_query: List[str],
        token_top_k: int = 100,
        document_top_k: int = 100,
        return_text: bool = True,
    ):
        """Runs XTR retrieval for a query."""
        batch_result = self.batch_search_tokens(batch_query, token_top_k=token_top_k)
        batch_mae = self.estimate_missing_similarity(batch_result)
        batch_ranking = self.aggregate_scores(batch_result, batch_mae, document_top_k)
        if return_text:
            return self.get_document_text(batch_ranking), batch_result
        else:
            return batch_ranking, batch_result
