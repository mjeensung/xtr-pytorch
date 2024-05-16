# coding=utf-8
# Copyright 2024 Jinhyuk Lee and Mujeen Sung. All rights reserved.
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
""" PyTorch XTR model. """




import math
import os

import torch
import torch.utils.checkpoint
from torch import nn
from typing import Optional, Tuple, Union

from transformers.modeling_utils import PreTrainedModel
import torch.nn.functional as F

from .configuration_xtr import XtrConfig
from transformers.utils import logging
from .utils import load_file_path

logger = logging.get_logger(__name__)

class XtrPreTrainedModel(PreTrainedModel):
    """
    A simple interface for downloading and loading pretrained models.
    """

    config_class = XtrConfig
    base_model_prefix = "xtr"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

class XtrModel(XtrPreTrainedModel):
    """
    """
    def __init__(self, model_name_or_path=None, config=None, device='cpu'):
        super().__init__(config)
        self.config = config
        self.t5_encoder = self._load_t5_encoder(model_name_or_path, device)
        self.linear_layer = self._load_linear_layer(model_name_or_path, device)
        
    def _load_t5_encoder(self,model_name_or_path=None, device='cpu'):
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        
        t5_encoder = T5EncoderModel.from_pretrained(model_name_or_path, use_safetensors=True).to(device=device)
        
        return t5_encoder

    def _load_linear_layer(self,model_name_or_path=None, device='cpu'):
        linear_path = load_file_path(
            model_name_or_path, "2_Dense/pytorch_model.bin"
        )
        linear_weight = torch.load(linear_path)
        linear_layer = torch.nn.Linear(self.config.in_features, self.config.out_features)
        linear_layer.weight = torch.nn.Parameter(linear_weight['linear.weight'].to(device=device))
        linear_layer.bias = torch.nn.Parameter(torch.zeros(self.config.out_features, device=device))
        return linear_layer

    def get_token_embed_dim(self):
        return self.config.out_features
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        def pass_through(model_output, attention_mask):
            token_embeddings = model_output[0] # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return token_embeddings * input_mask_expanded

        model_output = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Perform pooling
        embeddings = pass_through(model_output, attention_mask)

        # Apply linear layer
        embeddings = self.linear_layer(embeddings)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=2)

        return embeddings