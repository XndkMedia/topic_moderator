from torch import nn
from transformers import AutoModel
import torch

DROPOUT = 0.1


class EmotionClassifier(nn.Module):

    def __init__(self):
        super(EmotionClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained("ai-forever/ruBert-large")
        self.dropout = nn.Dropout(DROPOUT)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1024, 768)
        self.linear2 = nn.Linear(768, 6)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.linear(pooled_output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output

