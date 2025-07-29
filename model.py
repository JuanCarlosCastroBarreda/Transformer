import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, dim_model=64, num_heads=4, num_layers=2, num_classes=10):
        super(SimpleTransformer, self).__init__()

        self.flatten = nn.Flatten()
        self.input_proj = nn.Linear(28 * 28, dim_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(dim_model, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)
