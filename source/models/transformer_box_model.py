from transformers import BertModel, BertTokenizer
from torch import nn as nn
from transformers import AutoConfig
from box_wrapper import CenterSigmoidBoxTensor
from modules.highway_network import HighwayNetwork
from utils.constants import *


class TransformerBoxModel(nn.Module):
    def __init__(self):
        super(TransformerBoxModel, self).__init__()
        self.transformer_tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)
        self.transformer_config = AutoConfig.from_pretrained(MODEL_TYPE)
        self.encoder = BertModel.from_pretrained(MODEL_TYPE)
        self.proj_layer = HighwayNetwork(
            self.transformer_config.hidden_size,
            BOX_DIMENSION * 2,
            NUMBER_OF_PROJECTION_LAYERS,
            activation=nn.ReLU())

    def build_representation(self, inputs):
        mention_context_rep = self.encoder(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            output_hidden_states=False)
        mention_context_rep = mention_context_rep[0][:, 0, :]
        mention_context_rep = self.proj_layer(mention_context_rep)
        mention_context_rep = CenterSigmoidBoxTensor.from_split(mention_context_rep)
        return mention_context_rep
