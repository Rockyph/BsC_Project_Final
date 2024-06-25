import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Perceiver import TransformerBlock, SelfAttention
import einops

# VisionTransformer Neural Network
class VisionTransformer(nn.Module):
    def __init__(self, device, channels: int, image_size: int, batch_size: int,
        patch_size: int, embedding_size: int, attention_heads: int, ff: int,
        depth: int, nr_classes: int) -> None:
        """
        Function that initializes the Transformer class.
        Parameters: 
        - device (gpu or cuda)
        - image channels (should be 3 for RGB)
        - image size (should be 32)
        - batch size
        - patch size 
        - embedding size for representing patches
        - number of attention heads
        - constant to use in feedforward network in transformer block
        - number of transformer blocks
        - number of classes (should be 2 for real and fake)
        Layers:
        - Patch embeddings
        - Position embeddings
        - Linear layer for unifying the patched input and position embeddings
        - Sequence of transformer blocks
        - Linear layer for assigning class values
        """

        # Initialization
        super().__init__()

        # Parameters to use in forward call
        self.device = device
        self.patch_size = patch_size

        # Layers
        self.patch_embeddings = nn.Linear(channels * (patch_size ** 2), embedding_size)
        self.position_embeddings = nn.Embedding((image_size // patch_size) ** 2, embedding_size)
        self.unify_embeddings = nn.Linear(2 * embedding_size, embedding_size)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(attention_heads, embedding_size, ff) for block in range(depth)])
        self.classes = nn.Linear(embedding_size, nr_classes)

    def forward(self, batch: list) -> list:
        """
        Forward call of the Transformer class.
        Parameter: batch of images (batch of images (sequences of patches in 3 channels))
        Operations:
        - Split images from the batch in flattened patches
        - Apply layers defined in init
        - Apply a global average operation before applying the last layer
        Returns: output of transformer network
        """
        batch = einops.rearrange(batch, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        patch_emb = self.patch_embeddings(batch)
        x, y, z = patch_emb.size()
        pos_emb = self.position_embeddings(torch.arange(y, device=self.device))[None, :, :].expand(x, y, z)
        batch = self.unify_embeddings(torch.cat((patch_emb, pos_emb), dim=2).view(-1, 2 * z)).view(x, y, z)
        output = self.transformer_blocks(batch)
        output = torch.mean(output, dim=1)
        output = self.classes(output)
        return output