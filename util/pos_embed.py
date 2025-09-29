

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: (grid_height, grid_width)
    return:
    pos_embed: [grid_height*grid_width, embed_dim] or [1+grid_height*grid_width, embed_dim] (w/o or w/ cls_token)
    """
    grid_height, grid_width = grid_size
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h and grid_w, respectively
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid width
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_w = np.arange(grid_size, dtype=np.float32)

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_w)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed
    

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed_x(model, checkpoint_model):
    if 'pos_embed_x' in checkpoint_model:
        print("Loading pos_embed_x from checkpoint")
        # (1, 1+T'_max, D/2)
        pos_embed_checkpoint = checkpoint_model['pos_embed_x']

        # D/2
        embedding_size = pos_embed_checkpoint.shape[-1]
        
        # (1, 1, D/2)
        cls_embedding = pos_embed_checkpoint[:, :1]

        num_tokens_ckpt = pos_embed_checkpoint[:, 1:].shape[1]  # T'_max
        num_tokens_model = model.max_num_patches_x              # T'_model

        if num_tokens_model > num_tokens_ckpt:
            print("Position interpolate from %dx%d to %dx%d" % (1, num_tokens_ckpt, 1, num_tokens_model))
            # (1, T'_max, D/2)
            pos_tokens = pos_embed_checkpoint[:, 1:]
            # (1, D/2, T'_max)
            pos_tokens = pos_tokens.permute(0, 2, 1)
            # (1, D/2, T'_model)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=num_tokens_model, mode="linear", align_corners=False)
            # (1, T'_model, D/2)
            pos_tokens = pos_tokens.permute(0, 2, 1)
            # (1, 1+T'_model, D/2)
            new_pos_embed = torch.cat((cls_embedding, pos_tokens), dim=1)
            checkpoint_model['pos_embed_x'] = new_pos_embed
    
            model.pos_embed_x = torch.nn.Parameter(torch.zeros(1, num_tokens_model + 1, embedding_size), 
                                             requires_grad=False)
        else:
            model.pos_embed_x = torch.nn.Parameter(torch.zeros(1, num_tokens_ckpt + 1, embedding_size), 
                                             requires_grad=False)

        model.pos_embed_x.data.copy_(checkpoint_model['pos_embed_x'])
    else:
        print("Initializing new pos_embed_x")


def interpolate_decoder_pos_embed_x(model, checkpoint_model):
    if 'decoder_pos_embed_x' in checkpoint_model:
        print("Loading decoder_pos_embed_x from checkpoint")
        # (1, 1+T'_max, D_dec/2)
        pos_embed_checkpoint = checkpoint_model['decoder_pos_embed_x']

        # D_dec/2
        embedding_size = pos_embed_checkpoint.shape[-1]
        
        # (1, 1, D_dec/2)
        cls_embedding = pos_embed_checkpoint[:, :1]

        num_tokens_ckpt = pos_embed_checkpoint[:, 1:].shape[1]  # T'_max
        num_tokens_model = model.max_num_patches_x              # T'_model

        if num_tokens_model > num_tokens_ckpt:
            print("Position interpolate from %dx%d to %dx%d" % (1, num_tokens_ckpt, 1, num_tokens_model))
            # (1, T'_max, D_dec/2)
            pos_tokens = pos_embed_checkpoint[:, 1:]
            # (1, D_dec/2, T'_max)
            pos_tokens = pos_tokens.permute(0, 2, 1)
            # (1, D_dec/2, T'_model)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=num_tokens_model, mode="linear", align_corners=False)
            # (1, T'_model, D_dec/2)
            pos_tokens = pos_tokens.permute(0, 2, 1)
            # (1, 1+T'_model, D_dec/2)
            new_pos_embed = torch.cat((cls_embedding, pos_tokens), dim=1)
            checkpoint_model['decoder_pos_embed_x'] = new_pos_embed
    
            model.decoder_pos_embed_x = torch.nn.Parameter(torch.zeros(1, num_tokens_model + 1, embedding_size), 
                                             requires_grad=False)
        else:
            model.decoder_pos_embed_x = torch.nn.Parameter(torch.zeros(1, num_tokens_ckpt + 1, embedding_size), 
                                             requires_grad=False)

        model.decoder_pos_embed_x.data.copy_(checkpoint_model['decoder_pos_embed_x'])
    else:
        print("Initializing new decoder_pos_embed_x")