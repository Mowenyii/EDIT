from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import os
from ldm.modules.diffusionmodules.util import checkpoint
from collections import defaultdict
import numpy as np

rank=defaultdict(list)
lamb=0.0
is_add_maps=False
is_add_rank=False

def add_maps():
    global is_add_maps
    is_add_maps=True

def dont_add_maps():
    global is_add_maps
    is_add_maps=False

def add_rank():
    global is_add_rank
    is_add_rank=True

def dont_add_rank():
    global is_add_rank
    is_add_rank=False

def get_lamb():
    global lamb
    return lamb

def get_rank():
    global rank
    return rank

def edit_rank(r):
    global rank
    rank=r

def edit_lamb(b):
    global lamb
    lamb=b

def clear_lamb():
    global lamb

def clear_rank():
    global rank
    rank = defaultdict(list)

heat_maps = defaultdict(list)# 77, (16,64,64)
all_heat_maps = []


def clear_heat_maps():
    global heat_maps, all_heat_maps
    heat_maps = defaultdict(list)
    all_heat_maps = []


def next_heat_map():
    global heat_maps, all_heat_maps
    all_heat_maps.append(heat_maps)
    heat_maps = defaultdict(list)


def get_global_heat_map(last_n: int = None, idx: int = None, factors=None):
    global heat_maps, all_heat_maps

    if idx is not None:
        heat_maps2 = [all_heat_maps[idx]]
    else:
        heat_maps2 = all_heat_maps[-last_n:] if last_n is not None else all_heat_maps

    if factors is None:
        factors = {1, 2, 4, 8, 16, 32}

    all_merges = []

    for heat_map_map in heat_maps2: #heat_map_map是一个时间步所有尺寸的att map
        merge_list = []

        for k, v in heat_map_map.items():
            if k in factors:
                merge_list.append(torch.stack(v, 0).mean(0))
        if merge_list != []:
            all_merges.append(merge_list)
    #[10, 4, 77, 16, 64, 64]
    maps = torch.stack([torch.stack(x, 0) for x in all_merges], dim=0)
    return maps.sum(0).cuda().sum(2).sum(0)
    #sum(2)这里把16那维全sum了，包括un-condition
def get_global_heat_map_pic(last_n: int = None, idx: int = None, factors=None):
    global heat_maps, all_heat_maps

    if idx is not None:
        heat_maps2 = [all_heat_maps[idx]]
    else:
        heat_maps2 = all_heat_maps[-last_n:] if last_n is not None else all_heat_maps

    if factors is None:
        factors = {1, 2, 4, 8, 16, 32}

    all_merges = []

    for heat_map_map in heat_maps2: #heat_map_map是一个时间步所有尺寸的att map
        merge_list = []

        for k, v in heat_map_map.items():
            if k in factors:
                merge_list.append(torch.stack(v, 0).mean(0))
        if merge_list != []:
            all_merges.append(merge_list)
    #[10, 4, 77, 16, 64, 64]#[20, 4, 77, 24, 64, 64]
    maps = torch.stack([torch.stack(x, 0) for x in all_merges], dim=0)
    return maps


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., struct_attn=False, save_map=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.struct_attn = struct_attn
        self.save_map = save_map
    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    @torch.no_grad()
    def _up_sample_attn(self, x, factor, method: str = 'bicubic'):
        weight = torch.full((factor, factor), 1 / factor ** 2, device=x.device)
        weight = weight.view(1, 1, factor, factor)

        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.unsqueeze(1).view(map_.size(0), 1, h, w)
                if method == 'bicubic':
                    map_ = F.interpolate(map_, size=(64, 64), mode="bicubic", align_corners=False)
                    maps.append(map_.squeeze(1))
                else:
                    maps.append(F.conv_transpose2d(map_, weight, stride=factor).squeeze(1).cpu())

        maps = torch.stack(maps, 0).cpu()
        return maps

    def _attention(self, query, key, value, sequence_length, dim, use_context: bool = True):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                    torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
            )
            factor = int(math.sqrt(4096 // attn_slice.shape[1]))
            if use_context:
                global heat_maps, is_add_maps,rank,is_add_rank

            rank=get_rank()

            if use_context and rank != {} and attn_slice.shape[1] == 4096 and is_add_rank:
                e=50
                for i in range(len(rank)):
                    k_l=list(rank.keys())
                    if rank[k_l[i]] !=[]:
                        w=rank[k_l[i]][0]
                        attn_slice[:,:,k_l[i]]=attn_slice[:,:,k_l[i]]+e*w*rank[k_l[i]][1].flatten()

            attn_slice = attn_slice.softmax(-1)

            if use_context and is_add_maps:
                if factor >= 1:
                    factor //= 1
                    maps = self._up_sample_attn(attn_slice, factor)
                    global heat_maps
                    heat_maps[factor].append(maps)



            attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forward(self, x, context=None, scontext=None, pmask=None, time=None, mask=None):
        h = self.heads
        q = self.to_q(x)

        if isinstance(context, list):
            if self.struct_attn:
                out = self.struct_qkv(q, context, mask)
            else:
                context = torch.cat([context[0], context[1]['k'][0]], dim=0) # use key tensor for context
                out = self.normal_qkv(q, context, mask)
        else:
            if time==None:
                use_context = context is not None
                context = default(context, x)
                batch_size, sequence_length, dim = x.shape

                k = self.to_k(context)
                v = self.to_v(context)
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)
                # attention, what we cannot get enough of
                out = self._attention(q, k, v, sequence_length, dim, use_context=use_context)
                return self.to_out(out)
            if scontext == "selfattn":
                sim, attn, v = self.get_attmap(x=x, h=self.heads, context=context, mask=None)
                sattn = None
            else:
                if scontext is None:
                    use_context = context is not None
                    context = default(context, x)
                    batch_size, sequence_length, dim = x.shape

                    k = self.to_k(context)
                    v = self.to_v(context)
                    q = self.reshape_heads_to_batch_dim(q)
                    k = self.reshape_heads_to_batch_dim(k)
                    v = self.reshape_heads_to_batch_dim(v)
                    # attention, what we cannot get enough of
                    out = self._attention(q, k, v, sequence_length, dim, use_context=use_context)
                    return out, None

                else:

                    sim, attn, v = self.get_attmap(x=x, h=self.heads, context=context, mask=None)
                    ssim, sattn, sv = self.get_attmap(x=x, h=self.heads, context=scontext, mask=None)

                    """ cross attention control """
                    bh, hw, tleng = attn.shape
                    attn = self.cross_attention_control(tattmap=attn, sattmap=sattn, pmask=pmask, t=time, token_idx=[0],
                                                        weights=[[1., 1., 1.]])

            """ target prompt """
            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            out = self.to_out(out)

            if scontext != "selfattn":
                if scontext is not None:
                    """ source prompt """
                    sout = einsum('b i j, b j d -> b i d', sattn, sv)
                    sout = rearrange(sout, '(b h) n d -> b n (h d)', h=h)
                    sout = self.to_out(sout)



            return out, sim


        return self.to_out(out)



    def struct_qkv(self, q, context, mask):
        """
        context: list of [uc, list of conditional context]
        """
        uc_context = context[0]
        context_k, context_v = context[1]['k'], context[1]['v']


        if isinstance(context_k, list) and isinstance(context_v, list):
            if len(context)==2:
                out = self.multi_qkv(q, uc_context, context_k, context_v, mask)
            else:
                out = self.multi_qkv_2(q, uc_context, context_k, context_v, mask,context[-1])
        else:
            raise NotImplementedError
        return out

    def multi_qkv(self, q, uc_context, context_k, context_v, mask):
        h = self.heads

        assert uc_context.size(0) == context_k[0].size(0) == context_v[0].size(0)
        true_bs = uc_context.size(0) * h

        k_uc, v_uc = self.get_kv(uc_context)
        k_c = [self.to_k(c_k) for c_k in context_k]
        v_c = [self.to_v(c_v) for c_v in context_v]

        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

        k_uc = rearrange(k_uc, 'b n (h d) -> (b h) n d', h=h)
        v_uc = rearrange(v_uc, 'b n (h d) -> (b h) n d', h=h)

        k_c = [rearrange(k, 'b n (h d) -> (b h) n d', h=h) for k in k_c]  # NOTE: modification point
        v_c = [rearrange(v, 'b n (h d) -> (b h) n d', h=h) for v in v_c]

        # get composition
        sim_uc = einsum('b i d, b j d -> b i j', q[:true_bs], k_uc) * self.scale
        sim_c = [einsum('b i d, b j d -> b i j', q[true_bs:], k) * self.scale for k in k_c]

        attn_uc = sim_uc.softmax(dim=-1)
        attn_c = [sim.softmax(dim=-1) for sim in sim_c]

        if self.save_map and sim_uc.size(1) != sim_uc.size(2):
            self.save_attn_maps(attn_c)

        # get uc output
        out_uc = einsum('b i j, b j d -> b i d', attn_uc, v_uc)

        # get c output
        n_keys, n_values = len(k_c), len(v_c)


        if n_keys == n_values:
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn, v in zip(attn_c, v_c)]) / len(v_c)
        else:
            assert n_keys == 1 or n_values == 1
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn in attn_c for v in v_c]) / (n_keys * n_values)



        out = torch.cat([out_uc, out_c], dim=0)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return out

    def multi_qkv_2(self, q, uc_context, context_k, context_v, mask,src_con):
        h = self.heads

        assert uc_context.size(0) == context_k[0].size(0) == context_v[0].size(0)
        true_bs = uc_context.size(0) * h

        k_uc, v_uc = self.get_kv(uc_context)
        k_sc, v_sc = self.get_kv(src_con)
        k_c = [self.to_k(c_k) for c_k in context_k]
        v_c = [self.to_v(c_v) for c_v in context_v]

        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

        k_uc = rearrange(k_uc, 'b n (h d) -> (b h) n d', h=h)
        v_uc = rearrange(v_uc, 'b n (h d) -> (b h) n d', h=h)

        k_sc = rearrange(k_sc, 'b n (h d) -> (b h) n d', h=h)
        v_sc = rearrange(v_sc, 'b n (h d) -> (b h) n d', h=h)

        k_c = [rearrange(k, 'b n (h d) -> (b h) n d', h=h) for k in k_c]  # NOTE: modification point
        v_c = [rearrange(v, 'b n (h d) -> (b h) n d', h=h) for v in v_c]

        # get composition
        sim_uc = einsum('b i d, b j d -> b i j', q[:true_bs], k_uc) * self.scale
        sim_sc = einsum('b i d, b j d -> b i j', q[2*true_bs:], k_sc) * self.scale
        sim_c = [einsum('b i d, b j d -> b i j', q[true_bs:2*true_bs], k) * self.scale for k in k_c]

        attn_uc = sim_uc.softmax(dim=-1)
        attn_sc = sim_sc.softmax(dim=-1)
        attn_c = [sim.softmax(dim=-1) for sim in sim_c]

        if self.save_map and sim_uc.size(1) != sim_uc.size(2):
            self.save_attn_maps(attn_c)

        # get uc output
        out_uc = einsum('b i j, b j d -> b i d', attn_uc, v_uc)

        # get ec output
        out_sc = einsum('b i j, b j d -> b i d', attn_sc, v_sc)

        # get c output
        n_keys, n_values = len(k_c), len(v_c)

        rank = get_rank()

        if n_keys == n_values:
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn, v in zip(attn_c, v_c)]) / len(v_c)
        else:
            assert n_keys == 1 or n_values == 1
            if rank !={}:
                out_c = sum([einsum('b i j, b j d -> b i d', attn_c[a], v)*rank[0][a] for a in range(len(attn_c)) for v in v_c])
            else:
                out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn in attn_c for v in v_c]) / (
                            n_keys * n_values)
            # [einsum('b i j, b j d -> b i d', attn, v) for attn in attn_c for v in v_c]


        out = torch.cat([out_uc, out_c,out_sc], dim=0)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return out

    def normal_qkv(self, q, context, mask):
        h = self.heads
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        if self.save_map and sim.size(1) != sim.size(2):
            self.save_attn_maps(attn.chunk(2)[1])

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return out

    def get_kv(self, context):
        return self.to_k(context), self.to_v(context)

    def save_attn_maps(self, attn):
        h = self.heads
        if isinstance(attn, list):
            height = width = int(math.sqrt(attn[0].size(1)))
            self.attn_maps = [
                rearrange(m.detach(), '(b x) (h w) l -> b x h w l', x=h, h=height, w=width)[..., :20].cpu() for m in
                attn]
        else:
            height = width = int(math.sqrt(attn.size(1)))
            self.attn_maps = rearrange(attn.detach(), '(b x) (h w) l -> b x h w l', x=h, h=height, w=width)[...,
                             :20].cpu()

    # prompt-to-prompt
    def get_attmap(self, x, h, context, mask=None):
        # h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        query, key, val = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # import pdb; pdb.set_trace()
        sim = einsum('b i d, b j d -> b i j', query, key) * self.scale
        # attention, what we cannot get enough of
        rear_sim = rearrange(sim, '(b h) n d -> b h n d', h=h)

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        return rear_sim, attn, val

    def cross_attention_control(self, tattmap, sattmap=None, pmask=None, t=0, tthres=0, token_idx=[0],
                                weights=[[1., 1., 1.]]):
        attn = tattmap
        sattn = sattmap

        h = 8
        bh, n, d = attn.shape

        if t >= tthres:
            """ 1. swap & ading new phrase """
            if sattmap is not None:
                bh, n, d = attn.shape
                pmask, sindices, indices = pmask
                pmask = pmask.view(1, 1, -1).repeat(bh, n, 1)
                attn = (1 - pmask) * attn[:, :, indices] + (pmask) * sattn[:, :, sindices]

            """ 2. reweighting """
            attn = rearrange(attn, '(b h) n d -> b h n d',
                             h=h)  # (6,8,4096,77) -> (img1(uc), img2(uc), img3(uc), img1(c), img2(c), img3(c))
            num_iter = bh // (h * 2)  #: 3
            for k in range(len(token_idx)):
                for i in range(num_iter):
                    attn[num_iter + i, :, :, token_idx[k]] *= weights[k][i]
            attn = rearrange(attn, 'b h n d -> (b h) n d', h=h)  # (6,8,4096,77)

        return attn


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, struct_attn=False, save_map=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout,
                                    struct_attn=struct_attn, save_map=save_map)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, scontext=None, pmask=None, time=None):
        return checkpoint(self._forward, (x, context,scontext,pmask,time), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, scontext=None, pmask=None, time=None):
        if time==None:
            x = self.attn1(self.norm1(x)) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
            return x
        else:
            x_att, self_attmap = self.attn1(self.norm1(x), scontext="selfattn", time=time) # + x (8, 1024, 320)
            x += x_att
            x_att, cross_attmap = self.attn2(self.norm2(x), context=context, scontext=scontext, pmask=pmask, time=time) # +x
            x += x_att # (8, 1024, 320)
            x = self.ff(self.norm3(x)) + x      # (8, 1024, 320)
            return x, self_attmap, cross_attmap


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, is_get_attn=False, attn_save_dir="./attenion_map_savedir",struct_attn=False, save_map=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, struct_attn=struct_attn, save_map=save_map)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        self.is_get_attn = is_get_attn
        if self.is_get_attn:
            self.attmap_save_dir = attn_save_dir
            #self.attmap_save_dir = "./attenmaps_bear/"
            os.makedirs( os.path.join(self.attmap_save_dir, "selfatt"), exist_ok=True )
            os.makedirs( os.path.join(self.attmap_save_dir, "crossatt"), exist_ok=True )

    def avg_attmap(self, attmap, token_idx=0):
        """
        num_sample(=batch_size) = 3
        uc,c = 2 #(unconditional, condiitonal)
        -> 3*2=6

        attmap.shape: similarity matrix.
        token_idx: index of token for visualizing, 77: [SOS, ...text..., EOS]
        """
        nsample2, head, hw, context_dim = attmap.shape

        # import pdb; pdb.set_trace()
        attmap_sm = F.softmax(attmap.float(),
                              dim=-1)  # F.softmax(torch.Tensor(attmap).float(), dim=-1) # (6, 8, hw, context_dim)
        att_map_sm = attmap_sm[nsample2 // 2:, :, :, :]  # (3, 8, hw, context_dim)
        att_map_mean = torch.mean(att_map_sm, dim=1)  # (3, hw, context_dim)

        b, hw, context_dim = att_map_mean.shape
        h = int(math.sqrt(hw))
        w = h

        return att_map_mean.view(b, h, w, context_dim)  # (3, h, w, context_dim)

    def forward(self, x, context=None, scontext=None, pmask=None, timestep_str=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        cross_attmap=None
        for block in self.transformer_blocks:
            if timestep_str!=None:
                time = int(timestep_str.split("_")[1].split("time")[1])
                x, self_attmap, cross_attmap = block(x, context=context, scontext=scontext, pmask=pmask, time=time)
            else:
                x = block(x, context=context)
            if self.is_get_attn:
                if cross_attmap!=None:
                    if scontext is not None:
                        """ save attention map """
                        cross_attmap, scross_attmap = cross_attmap.chunk(2)
                        #np.save( os.path.join(self.attmap_save_dir, "selfatt", timestep_str), self.avg_attmap(self_attmap).detach().cpu().numpy() )
                        np.save( os.path.join(self.attmap_save_dir, "crossatt", timestep_str), self.avg_attmap(cross_attmap).detach().cpu().numpy() )
                    else:
                        """ save attention map """
                        #np.save( os.path.join(self.attmap_save_dir, "selfatt", timestep_str), self.avg_attmap(self_attmap).detach().cpu().numpy() )
                        np.save( os.path.join(self.attmap_save_dir, "crossatt", timestep_str), self.avg_attmap(cross_attmap).detach().cpu().numpy() )


        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in