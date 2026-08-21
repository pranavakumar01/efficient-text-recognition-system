import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    """
    Deep CNN Feature Extractor (ResNet-style backbone) for line/word text image representations.
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 512):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # (H/2, W/2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # (H/4, W/4)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d((2, 1), (2, 1))  # Pool height only

        self.conv5 = nn.Conv2d(512, feature_dim, kernel_size=(4, 1), stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = F.relu(self.bn5(self.conv5(x)))
        # Reshape to sequence: [B, W, Feature_Dim]
        b, c, h, w = x.size()
        x = x.squeeze(2).permute(0, 2, 1)  # [B, W, C]
        return x

class BahdanauAttention(nn.Module):
    """
    Bahdanau Sequence Attention Mechanism.
    """
    def __init__(self, hidden_dim: int):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # query: [B, hidden_dim], values: [B, seq_len, hidden_dim]
        score = self.V(torch.tanh(self.W1(values) + self.W2(query).unsqueeze(1)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * values, dim=1)
        return context_vector, attention_weights

class CNN_BiLSTM_Attention(nn.Module):
    """
    Primary OCR Model: CNN + BiLSTM + Attention Decoder.
    Supports transfer learning initialization from SimCLR Self-Supervised pre-trained backbone.
    """
    def __init__(self, num_classes: int, hidden_dim: int = 256, in_channels: int = 1):
        super(CNN_BiLSTM_Attention, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(in_channels=in_channels, feature_dim=hidden_dim)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        self.attention = BahdanauAttention(hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def load_ssl_weights(self, ssl_checkpoint_path: str):
        """
        Loads pre-trained SimCLR contrastive representation weights into the CNN feature extractor.
        """
        if not os.path.exists(ssl_checkpoint_path):
            print(f"[!] SSL Checkpoint '{ssl_checkpoint_path}' not found. Using random initialization.")
            return False

        try:
            ckpt = torch.load(ssl_checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)

            mapping = {
                "encoder.0.weight": "feature_extractor.conv1.weight",
                "encoder.0.bias": "feature_extractor.conv1.bias",
                "encoder.1.weight": "feature_extractor.bn1.weight",
                "encoder.1.bias": "feature_extractor.bn1.bias",
                "encoder.1.running_mean": "feature_extractor.bn1.running_mean",
                "encoder.1.running_var": "feature_extractor.bn1.running_var",
                "encoder.4.weight": "feature_extractor.conv2.weight",
                "encoder.4.bias": "feature_extractor.conv2.bias",
                "encoder.5.weight": "feature_extractor.bn2.weight",
                "encoder.5.bias": "feature_extractor.bn2.bias",
                "encoder.5.running_mean": "feature_extractor.bn2.running_mean",
                "encoder.5.running_var": "feature_extractor.bn2.running_var",
            }

            model_dict = self.state_dict()
            loaded_count = 0
            for ssl_key, target_key in mapping.items():
                if ssl_key in state_dict and target_key in model_dict:
                    model_dict[target_key].copy_(state_dict[ssl_key])
                    loaded_count += 1

            self.load_state_dict(model_dict)
            print(f"[*] Successfully transferred {loaded_count} SSL pre-trained layer tensors from '{ssl_checkpoint_path}'")
            return True
        except Exception as e:
            print(f"[!] Error transferring SSL weights: {e}")
            return False

    def forward(self, x: torch.Tensor) -> tuple:
        # Dynamic height guard: ensure height is 32 for CNN pooling layers
        if x.size(2) != 32:
            x = F.interpolate(x, size=(32, max(16, int(x.size(3) * (32.0 / x.size(2))))), mode='bilinear', align_corners=False)

        features = self.feature_extractor(x)  # [B, Seq_Len, Hidden_Dim]
        lstm_out, _ = self.bilstm(features)   # [B, Seq_Len, Hidden_Dim * 2]
        
        # Compute Bahdanau sequence attention
        query = torch.mean(lstm_out, dim=1)
        context, att_weights = self.attention(query, lstm_out)

        # Attentive feature modulation: modulate sequence with temporal attention weights
        attended_seq = lstm_out * (1.0 + att_weights)
        logits = self.classifier(attended_seq)   # [B, Seq_Len, Num_Classes]
        return logits, att_weights
