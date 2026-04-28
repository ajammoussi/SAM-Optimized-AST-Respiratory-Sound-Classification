"""
CustomAST
---------
Audio Spectrogram Transformer fine-tuned for 4-class ICBHI classification.

Architecture is identical to the paper:
  • AST backbone pre-trained on AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593)
  • Mean-pooling over the sequence dimension (more stable than CLS token alone)
   • Classifier head: Dropout(0.3) to Linear(768 to num_classes)
"""

import torch.nn as nn
from transformers import ASTModel


class CustomAST(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.ast = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        # x : (B, freq_bins, time_frames)  — produced by ASTFeatureExtractor
        outputs = self.ast(x)
        # Mean-pool over all patch tokens (including CLS / distillation tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # (B, 768)
        return self.classifier(embeddings)                   # (B, num_classes)
