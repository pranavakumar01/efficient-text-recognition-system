import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR_SSL_Backbone(nn.Module):
    """
    Self-Supervised Learning (SSL) Feature Representation Module (SimCLR Contrastive Architecture).
    Allows learning representations from unlabeled scene/document text crops without full label dependency.
    """
    def __init__(self, feature_dim: int = 128):
        super(SimCLR_SSL_Backbone, self).__init__()
        # Encoder Backbone (CNN stem)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Projection Head for Contrastive Learning
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)

def info_nce_loss(features_i: torch.Tensor, features_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    InfoNCE Contrastive Loss function for self-supervised representation learning.
    """
    batch_size = features_i.size(0)
    features = torch.cat([features_i, features_j], dim=0)
    similarity_matrix = torch.matmul(features, features.T) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    positives = torch.exp(torch.sum(features_i * features_j, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)

    denominator = torch.sum(torch.exp(similarity_matrix), dim=1)
    loss = -torch.log(positives / denominator)
    return loss.mean()
