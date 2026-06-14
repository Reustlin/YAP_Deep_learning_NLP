import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class CaloriesModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Image encoder
        self.image_encoder = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        image_dim = self.image_encoder.num_features
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Mass projection
        self.mass_projection = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        # Regressor
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, input_ids, attention_mask, mass):
        img_feat = self.image_encoder(image)
        img_feat = self.image_projection(img_feat)

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0]   # [CLS] токен
        text_feat = self.text_projection(text_feat)

        mass = mass.unsqueeze(1)
        mass_feat = self.mass_projection(mass)

        features = torch.cat([img_feat, text_feat, mass_feat], dim=1)
        output = self.classifier(features)
        return output.squeeze(1)
