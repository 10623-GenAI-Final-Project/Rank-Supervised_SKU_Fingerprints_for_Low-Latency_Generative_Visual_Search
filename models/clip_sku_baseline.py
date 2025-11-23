# models/clip_sku_baseline.py

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipSkuBaseline(nn.Module):
    """
    CLIP/SigLIP-style bi-encoder + SKU embedding table.

    A pretrained CLIP model is passed in (from open_clip).
    We add an nn.Embedding(num_skus, D) and train it (and optionally unfreeze CLIP).
    """

    def __init__(
        self,
        clip_model: nn.Module,
        num_skus: int,
        freeze_towers: bool = True,
    ) -> None:
        super().__init__()

        self.clip_model = clip_model

        embed_dim = self.clip_model.text_projection.shape[1]

        self.sku_embed = nn.Embedding(num_skus, embed_dim)
        nn.init.normal_(self.sku_embed.weight, std=0.02)

        self.freeze_towers = freeze_towers
        if self.freeze_towers:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Trainable temperature for InfoNCE.
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    @property
    def embed_dim(self) -> int:
        return self.clip_model.text_projection.shape[1]

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            embs: (B, D) L2-normalized
        """
        feats = self.clip_model.encode_image(images)
        feats = F.normalize(feats, dim=-1)
        return feats

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens: (B, L)
        Returns:
            embs: (B, D) L2-normalized
        """
        feats = self.clip_model.encode_text(text_tokens)
        feats = F.normalize(feats, dim=-1)
        return feats

    def sku_embeddings(self) -> torch.Tensor:
        """
        Returns:
            (num_skus, D) L2-normalized SKU embeddings.
        """
        w = self.sku_embed.weight
        w = F.normalize(w, dim=-1)
        return w

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward used for training.

        Returns:
            img_emb: (B, D)
            txt_emb: (B, D)
            sku_emb_all: (Ns, D)
            logit_scale: scalar exp(temperature)
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(text_tokens)
        sku_emb_all = self.sku_embeddings()
        logit_scale = self.logit_scale.exp()
        return img_emb, txt_emb, sku_emb_all, logit_scale
