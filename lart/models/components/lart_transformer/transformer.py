import math

import torch
from einops import rearrange
from torch import nn

from phalp.models.heads.smpl_head import SMPLHead
from phalp.configs.base import FullConfig



def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe          = torch.zeros(length, d_model)
    position    = torch.arange(0, length).unsqueeze(1)
    div_term    = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim    = dim_head *  heads
        project_out  = not (heads == 1 and dim_head == dim)

        self.heads   = heads
        self.scale   = dim_head ** -0.5
        self.attend  = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out  = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask_all):
        qkv          = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v      = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots         = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        masks_np     = mask_all[0]
        masks_bert   = mask_all[1]
        BS           = masks_np.shape[0]
        masks_np     = masks_np.view(BS, -1)
        masks_bert   = masks_bert.view(BS, -1)
        
        masks_np_    = rearrange(masks_np, 'b i -> b () i ()') * rearrange(masks_np, 'b j -> b () () j')
        masks_np_    = masks_np_.repeat(1, self.heads, 1, 1)
        
        masks_bert_  = rearrange(masks_bert, 'b i -> b () () i')
        masks_bert_  = masks_bert_.repeat(1, self.heads, masks_bert_.shape[-1], 1)
                
        dots[masks_np_==0]   = -1e3
        dots[masks_bert_==1] = -1e3
        
        del masks_np, masks_np_, masks_bert, masks_bert_
        
        attn    = self.attend(dots)
        attn    = self.dropout(attn)

        out     = torch.matmul(attn, v)
        out     = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., drop_path = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask_np):
        for attn, ff in self.layers:
            x_          = attn(x, mask_all=mask_np) 
            x           = x + self.drop_path(x_)
            x           = x + self.drop_path(ff(x))
            
        return x

class lart_transformer(nn.Module):
    def __init__(self, opt, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., droppath = 0., device=None):
        super().__init__()
        self.cfg  = opt
        self.dim  = dim
        self.device = device
        self.mask_token = nn.Parameter(torch.randn(self.dim,))
        self.class_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        self.pos_embedding = nn.Parameter(positionalencoding2d(self.dim, 250, 10))#.to(self.device)
        self.register_buffer('pe', self.pos_embedding)
        
        self.transformer    = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout, drop_path = droppath)

        pad                 = self.cfg.transformer.conv.pad
        stride              = self.cfg.transformer.conv.stride
        kernel              = stride + 2 * pad
        self.conv_en        = nn.Conv1d(self.dim, self.dim, kernel_size=kernel, stride=stride, padding=pad)
        self.conv_de        = nn.ConvTranspose1d(self.dim, self.dim, kernel_size=kernel, stride=stride, padding=pad)

        # Pose shape encoder for encoding pose shape features, used by default
        self.pose_shape_encoder     = nn.Sequential(
                                            nn.Linear(self.cfg.extra_feat.pose_shape.dim, self.cfg.extra_feat.pose_shape.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.cfg.extra_feat.pose_shape.mid_dim, self.cfg.extra_feat.pose_shape.en_dim),
                                        )
        
        # SMPL head for predicting SMPL parameters
        phalp_config                = FullConfig()
        self.smpl_head              = nn.ModuleList([SMPLHead(phalp_config, input_dim=self.cfg.in_feat, pool='pooled') for _ in range(self.cfg.num_smpl_heads)])
        
        # Location head for predicting 3D location of the person
        self.loca_head              = nn.ModuleList([nn.Sequential(
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat), 
                                            nn.ReLU(), 
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat),
                                            nn.ReLU(),         
                                            nn.Linear(self.cfg.in_feat, 3)
                                        ) for _ in range(self.cfg.num_smpl_heads)])

        # Action head for predicting action class in AVA dataset labels  
        ava_action_classes          = self.cfg.ava.num_action_classes if not(self.cfg.ava.predict_valid) else self.cfg.ava.num_valid_action_classes
        self.action_head_ava        = nn.ModuleList([nn.Sequential(    
                                            nn.Linear(self.cfg.in_feat, ava_action_classes),
                                        ) for _ in range(self.cfg.num_smpl_heads)])
        
        # Action head for predicting action class in Kinetics dataset labels
        self.action_head_kinetics   = nn.Sequential(
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat), 
                                            nn.ReLU(), 
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat),
                                            nn.ReLU(),         
                                            nn.Linear(self.cfg.in_feat, self.cfg.kinetics.num_action_classes),
                                        )
        
        # Joints encoder for encoding 3D joints
        if("joints_3D" in self.cfg.extra_feat.enable):
            self.joint_encoder       = nn.Sequential(
                                            nn.Linear(self.cfg.extra_feat.joints_3D.dim, self.cfg.extra_feat.joints_3D.mid_dim), nn.ReLU(), 
                                            nn.Linear(self.cfg.extra_feat.joints_3D.mid_dim, self.cfg.extra_feat.joints_3D.mid_dim), nn.ReLU(),         
                                            nn.Linear(self.cfg.extra_feat.joints_3D.mid_dim, self.cfg.extra_feat.joints_3D.en_dim),
                                        )
        
        # apperance encoder for encoding apperance/pixels features
        if("apperance" in self.cfg.extra_feat.enable):
            self.apperance_encoder       = nn.Sequential(
                                            nn.Linear(self.cfg.extra_feat.apperance.dim, self.cfg.extra_feat.apperance.mid_dim), nn.ReLU(), 
                                            nn.Linear(self.cfg.extra_feat.apperance.mid_dim, self.cfg.extra_feat.apperance.mid_dim), nn.ReLU(),         
                                            nn.Linear(self.cfg.extra_feat.apperance.mid_dim, self.cfg.extra_feat.apperance.en_dim),
                                        )
                    
    def bert_mask(self, data, mask_type):
        if(mask_type=="random"):
            has_detection  = data['has_detection']==1
            mask_detection = data['mask_detection']
            for i in range(data['has_detection'].shape[0]):
                indexes        = has_detection[i].nonzero()
                indexes_mask   = indexes[torch.randperm(indexes.shape[0])[:int(indexes.shape[0]*self.cfg.mask_ratio)]]
                mask_detection[i, indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2]] = 1.0
        elif(mask_type=="zero"):
            dtype = data['has_detection'].dtype
            has_detection  = data['has_detection']==0
            mask_detection = data['mask_detection']
            indexes_mask   = has_detection.nonzero()
            mask_detection[indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2], :] = 1.0
            has_detection = has_detection*0 + 1.0
            mask_detection = mask_detection.to(dtype)
            has_detection  = has_detection.to(dtype)
        else:
            raise NotImplementedError

        return data, has_detection, mask_detection

    def forward(self, data, mask_type="random"):
        
        # prepare the input data and masking
        dtype = data['pose_shape'].dtype
        data, has_detection, mask_detection = self.bert_mask(data, mask_type)

        # encode the input pose tokens
        pose_   = data['pose_shape'].to(dtype)
        pose_en = self.pose_shape_encoder(pose_)
        x       = pose_en

        if("joints_3D" in self.cfg.extra_feat.enable):          
            joint_ = data['joints_3D'].to(dtype)
            joint_en = self.joint_encoder(joint_)            
            x = torch.cat((x, joint_en), dim=-1)

        if("apperance" in self.cfg.extra_feat.enable):
            apperance_ = data['apperance_emb'].to(dtype)
            apperance_en = self.apperance_encoder(apperance_)
            x = torch.cat((x, apperance_en), dim=-1)
        
        # mask the input tokens
        x[mask_detection[:, :, :, 0]==1] = self.mask_token

        BS, T, P, dim = x.size()
        x = x.view(BS, T*P, dim)
        loss = torch.zeros(1).to(x.device)

        # adding 2D posistion embedding
        x = x + self.pos_embedding[None, :, :self.cfg.frame_length, :self.cfg.max_people].reshape(1, dim, self.cfg.frame_length*self.cfg.max_people).permute(0, 2, 1)
        
        x = self.transformer(x, [has_detection, mask_detection])
        x = torch.concat([self.class_token.repeat(BS, self.cfg.max_people, 1), x], dim=1)
            
        return x, loss