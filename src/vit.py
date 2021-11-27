from transformers import ViTModel, ViTFeatureExtractor
import torch
from torch import nn
import numpy as np


class ViT(nn.Module):
    def __init__(self, num_labels=10):
        super(ViT, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.num_labels = num_labels
        self.loss = nn.CrossEntropyLoss()

    def forward(self, image_batch, labels):

        # SRC: https://github.com/395t/coding-assignment-week-8-vit-1/blob/main/notebooks/ViT.ipynb
        x = np.split(np.squeeze(np.array(image_batch)), len(image_batch))
        # Remove unecessary dimension
        for index, array in enumerate(x):
          x[index] = np.squeeze(array)
        # Apply feature extractor, stack back into 1 tensor and then convert to tensor
        images = torch.tensor(np.stack(self.feature_extractor(x)['pixel_values'], axis=0))

        outputs = self.model(images)
        output = outputs.last_hidden_state[:,0]
        logits = self.classifier(output)
        # loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        loss = self.loss(logits, labels)
        pred = torch.argmax(logits, dim=1)
        return loss, logits, pred

