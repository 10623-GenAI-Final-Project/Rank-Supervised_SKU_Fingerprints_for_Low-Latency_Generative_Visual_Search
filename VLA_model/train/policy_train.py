import torch
from torch.utils.data import DataLoader
from model.policy import VLAPolicy
from data.dataset import VLADataset
import torch.nn.functional as F
import torch.nn as nn


def train_policy(samples, visual_dim, quality_dim, num_actions, device="cuda"):
    dataset = VLADataset(samples)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = VLAPolicy(visual_dim, quality_dim, num_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total, correct, loss_sum = 0, 0, 0

        for vfeat, qfeat, labels in loader:
            vfeat, qfeat, labels = (
                vfeat.to(device),
                qfeat.to(device),
                labels.to(device),
            )

            logits = model(vfeat, qfeat)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * labels.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch+1}: Loss={loss_sum/total:.4f}, Acc={correct/total:.3f}"
        )

    return model
