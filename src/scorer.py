
import torch, torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

class TokenScorer(nn.Module):
    def __init__(self, hidden_dim=768, freeze_encoder=False):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return torch.sigmoid(self.head(h)).squeeze(-1)

def load_scorer(checkpoint="models/scorer.pt", device="cpu"):
    m = TokenScorer()
    m.load_state_dict(torch.load(checkpoint, map_location=device))
    return m.eval().to(device)
