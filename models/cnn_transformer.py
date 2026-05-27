import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class MultiScaleStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2, bias=False),
                nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3, bias=False),
                nn.Conv1d(in_channels, branch_channels, kernel_size=9, padding=4, bias=False),
            ]
        )
        self.project = nn.Sequential(
            nn.BatchNorm1d(branch_channels * len(self.branches)),
            nn.GELU(),
            nn.Conv1d(branch_channels * len(self.branches), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcite1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + residual)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, conv_kernel_size=9):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=d_model,
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        attn_input = self.attn_norm(x)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        conv_input = self.conv_norm(x).transpose(1, 2)
        x = x + self.conv(conv_input).transpose(1, 2)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class ConformerAMC(nn.Module):
    """
    Strong AMC backbone for RadioML-style IQ tensors.

    Input:  (batch, 2, 1024)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        num_classes=24,
        d_model=256,
        nhead=8,
        num_layers=4,
        input_channels=2,
        input_length=1024,
        dropout=0.15,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_length = input_length
        self.stem = MultiScaleStem(input_channels, 64)
        self.cnn = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout=dropout * 0.5),
            ResidualBlock(128, 128, stride=1, dilation=2, dropout=dropout * 0.5),
            ResidualBlock(128, d_model, stride=2, dropout=dropout * 0.5),
            ResidualBlock(d_model, d_model, stride=1, dilation=2, dropout=dropout * 0.5),
            ResidualBlock(d_model, d_model, stride=2, dropout=dropout * 0.5),
            ResidualBlock(d_model, d_model, stride=2, dropout=dropout * 0.5),
        )

        processed_length = max(1, input_length // 16)
        self.pos_embedding = nn.Parameter(torch.randn(1, processed_length, d_model) * 0.02)
        self.encoder = nn.Sequential(
            *[ConformerBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers)]
        )
        self.attention_pool = AttentionPooling(d_model)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.cnn(x).transpose(1, 2)
        if x.size(1) != self.pos_embedding.size(1):
            pos = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=x.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.pos_embedding
        x = self.encoder(x + pos)
        pooled = self.attention_pool(x)
        mean_pooled = x.mean(dim=1)
        return self.classifier_head(torch.cat([pooled, mean_pooled], dim=1))
