# models/text_module.py
import torch
import torch.nn as nn
from transformers import BertModel

class TextModule(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased", out_features=512):
        super(TextModule, self).__init__()
        self.bert = BertModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, out_features)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_output = self.dropout(cls_output)
        return self.fc(cls_output)  # [batch, out_features]
