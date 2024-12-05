!pip install timm
!pip install transformers

"""#### Installing required packages"""

import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt
from google.colab import files

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer


!pip install kaggle --upgrade
os.environ['KAGGLE_USERNAME'] = "udyan1"
os.environ['KAGGLE_KEY'] = "aaa522eb595fc307cf371f0bb75c5b2d"

### For Flickr 8k
!kaggle datasets download -d adityajn105/flickr8k
!unzip flickr8k.zip
dataset = "8k"

"""## Data Pre-Processing"""

import pandas as pd

# Read the captions from the text file
df = pd.read_csv("captions.txt")

# Add an 'id' column to identify each image-caption pair
df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]

# Save the DataFrame to a CSV file
df.to_csv("captions.csv", index=False)

# Read the CSV file back into a DataFrame
df = pd.read_csv("captions.csv")

# Define the paths for images and captions
image_path = "/content/Images"
captions_path = "/content"

# Display the first few rows of the DataFrame
df.head(10)

df.info()

"""## Config

By using the CFG class, we centralize and organize the configuration parameters, making it easier to manage and modify them. It also improves code readability and maintainability by providing a single source of truth for all configuration settings.
"""

from itertools import product

class CFG:
    # Debug mode
    debug = False

    # Paths to image and caption data
    image_path = image_path
    captions_path = captions_path

    # Data loading parameters
    batch_size = 32
    num_workers = 2

    # Learning rates
    #head_lr = 1e-3
    # image_encoder_lr = 1e-4
    # text_encoder_lr = 1e-5

    # Regularization
    # weight_decay = 1e-3

    # Learning rate scheduler
    patience = 1
    factor = 0.8

    # Training parameters
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    # Pre-training and fine-tuning options
    pretrained = True  # Use pre-trained weights for both image and text encoders
    trainable = True  # Fine-tune both image and text encoders
    temperature = 1.0

    # Image size
    size = 224

    # Projection head parameters (used for both image and text encoders)
    num_projection_layers = 1
    projection_dim = 256

    # Dropout rate for regularization
    dropout_rate = 0.1  # Dropout rate to prevent overfitting

    # Define hyperparameter grid
    hyperparameter_grid = {
        'head_lr': [5e-4, 1e-3],
        'image_encoder_lr': [1e-5, 1e-4],
        'text_encoder_lr': [1e-6, 1e-5],
        'weight_decay': [5e-3, 1e-2],
    }

    # Generate all combinations of hyperparameters
    grid_combinations = list(itertools.product(*hyperparameter_grid.values()))

## print grid combination
for combination in CFG.grid_combinations:
    print(combination)

"""## Utils

The AvgMeter class provides a convenient way to compute and track the average value of a metric, while the get_lr function retrieves the learning rate from an optimizer, which is useful for monitoring and logging during training. These utilities are commonly used in deep learning training loops to monitor model performance and optimizer behavior.
"""

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = 0, 0, 0

    def update(self, val, count=1):
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]



class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

"""## Image Encoder

The ImageEncoder class encapsulates an image encoder model that converts input images into fixed-size vector representations. During forward pass, it passes the input images through the model and returns the resulting fixed-size vectors. This class is often used as a component in multimodal models where both images and text inputs are processed jointly.
"""

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

"""## Text Encoder

The TextEncoder class utilizes a DistilBERT model to encode textual inputs. During the forward pass, it passes input token indices and attention masks through the DistilBERT model and extracts the hidden state representation of the [CLS] token, which is used as the sentence-level embedding. This class is commonly used in multimodal models where both images and text inputs are processed jointly.
"""

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

"""## Projection Head

The ProjectionHead class implements a projection head that reduces the dimensionality of input embeddings while preserving important information. It applies linear projection, activation, fully connected transformation, dropout, residual connection, and layer normalization to achieve this. The resulting lower-dimensional embeddings can be used as features for downstream tasks or as part of a larger model architecture, such as a contrastive learning framework.
"""

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout_rate
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

"""## CLIP

The CLIPModel class encapsulates the CLIP model architecture, including image and text encoders and projection heads. It computes contrastive loss based on the similarity between images and text embeddings projected into a common space. The cross_entropy function is used to calculate the cross-entropy loss, which is part of the contrastive loss computation.
"""

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

batch_size = 4
dim = 256
embeddings = torch.randn(batch_size, dim)
out = embeddings @ embeddings.T
print(F.softmax(out, dim=-1))


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter



import copy

def main(config,grid_combinations,hyperparameter_grid):
    # Use default CFG if no external configuration is provided
    base_config = config or CFG()

    # Prepare data
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(base_config.text_tokenizer)

    best_overall_loss = float('inf')
    best_overall_config = None

    for combination in grid_combinations:
        # Update configuration with current hyperparameters
        current_config = copy.deepcopy(base_config)
        for i, key in enumerate(hyperparameter_grid.keys()):
            setattr(current_config, key, combination[i])

        print(f"\nTesting configuration: {combination}")

        # Build data loaders with current batch size
        train_loader = build_loaders(train_df, tokenizer, mode="train")
        valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

        # Initialize model, optimizer, and scheduler with current hyperparameters
        model = CLIPModel().to(current_config.device)
        params = [
            {"params": model.image_encoder.parameters(), "lr": current_config.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": current_config.text_encoder_lr},
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": current_config.head_lr, "weight_decay": current_config.weight_decay},
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=current_config.patience, factor=current_config.factor
        )
        step = "epoch"

        # Training loop
        best_loss = float('inf')
        for epoch in range(current_config.epochs):
            print(f"\nEpoch {epoch + 1}/{current_config.epochs}")

            # Training phase
            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
            print(f"Training Loss: {train_loss.avg:.4f}")

            # Validation phase
            model.eval()
            with torch.no_grad():
                valid_loss = valid_epoch(model, valid_loader)
            print(f"Validation Loss: {valid_loss.avg:.4f}")

            # Save the best model for this configuration
            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(model.state_dict(), f"/content/best_model_config_{combination}.pt")
                print("Best Model Saved for Current Configuration!")

            # Adjust learning rate
            lr_scheduler.step(valid_loss.avg)

        print(f"\nConfiguration {combination} Complete! Best Validation Loss: {best_loss:.4f}")

        # Update best overall configuration if necessary
        if best_loss < best_overall_loss:
            best_overall_loss = best_loss
            best_overall_config = combination

    print(f"\nGrid Search Complete!")
    print(f"Best Overall Configuration: {best_overall_config}")
    print(f"Best Overall Validation Loss: {best_overall_loss:.4f}")
