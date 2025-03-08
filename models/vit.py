import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functions.visualize import plot_grid_samples_tensor


class PatchEmbedding(nn.Module):
    def __init__(self, model_settings):
        """ 
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = (model_settings["patch_size"], model_settings["patch_size"])
        self.num_channels = model_settings["num_channels"]
        self.embedding_size = model_settings["embedding_size"]
        
        self.num_patches = np.prod(model_settings["input_shape"]) // np.prod(self.patch_size)

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size[0])
        self.linear1 = nn.Linear(np.prod(self.patch_size) * self.num_channels, self.embedding_size)

    def forward(self, x):
        #TODO: test with channels > 1
        x = self.unfold(x)          # Extract patches
        x = torch.movedim(x, 1,-1)  # Move patch pixels to last dim
        x = self.linear1(x)         # Create embeddings from patches
        
        # Add extra empty embedding dim, so we can sum class dim to this later on
        zeros = torch.zeros(x.shape[0], self.embedding_size, device=x.device).unsqueeze(1)
        x = torch.cat([x,zeros], dim=1)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, patch_embeddings, model_settings):
        super(PositionalEmbedding, self).__init__()
        self.num_embeddings = patch_embeddings + 1 # We add one extra embeddings, not used for position, but as the class embedding
        self.embedding_dim = model_settings["embedding_size"]
        self.device = model_settings["device"]

        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim, device=self.device)

    def forward(self, x):
        # We add one extra embeddings, not used for position, but as the class embedding
        positional_ints = torch.arange(0, self.num_embeddings, requires_grad=False, device=self.device
                                       ).repeat(x.shape[0], 1)
        embedding = self.embedding(positional_ints)
        return embedding


class TransformerBlock(nn.Module):
    def __init__(self, model_settings, dropout=0):
        """ 
        """
        super(TransformerBlock, self).__init__()
        self.embedding_dim = model_settings["embedding_size"]
        self.heads = model_settings["attention_heads"]
        hidden_dim = self.embedding_dim * 2

        self.layer_norm_1 = nn.LayerNorm(self.embedding_dim)
        self.attention = nn.MultiheadAttention(self.embedding_dim, self.heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_dim)

        self.linear = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.embedding_dim),
            nn.Dropout(dropout))
        print("test end")


    def forward(self, x_in):
        x = self.layer_norm_1(x_in)
        x = self.attention(x, x, x)
        print(len(x))

        return x
        


class VIT(nn.Module):
    """ """

    def __init__(self, model_settings):
        """ 
        """
        super(VIT, self).__init__()
        self.num_channels = model_settings["num_channels"]
        self.input_shape = model_settings["input_shape"]
        self.embedding_size = model_settings["embedding_size"]
        self.device = model_settings["device"]

        self.patch_embedding = PatchEmbedding(model_settings)
        self.positional_embedding = PositionalEmbedding(self.patch_embedding.num_patches, model_settings)

        self.transformer = TransformerBlock(model_settings)


    def forward(self, x):

        x = self.patch_embedding(x) 
        pos_embeddings = self.positional_embedding(x) # Also includes extra embeddings for class tokens
        x = x + pos_embeddings
        print("test begin")
        print(x.shape)
        x = self.transformer(x)

        return x