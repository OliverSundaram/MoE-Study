import torch
from torch import nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast



class Norm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(cfg["emb_dim"]))
        self.shift = nn.Parameter(torch.zeros(cfg["emb_dim"]))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift



class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["hidden_dim"]),
            nn.GELU(),
            nn.Linear(cfg["hidden_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)



class MoE(nn.Module):

    def __init__(self, cfg: dict[str, int | bool]):
        super().__init__()
        self.n_experts = cfg["n_experts"]
        self.top_k = cfg["top_k"]
        self.experts = nn.ModuleList(
            [FeedForward(cfg) for _ in range(self.n_experts)]
        )
        self.router = nn.Linear(cfg["emb_dim"], self.n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, emb_dim = x.shape
        tokens = x.reshape(batch_size * seq_len, emb_dim)

        router_logits = self.router(tokens)
        router_probs = torch.softmax(router_logits, dim=-1)

        top_weights, top_experts = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        expert_mask = F.one_hot(top_experts, self.n_experts)

        tokens_per_expert = torch.sum(expert_mask, dim=1).float().mean(dim=0) / self.top_k
        prob_per_expert = router_probs.mean(dim=0)

        aux_loss = self.n_experts * torch.sum(tokens_per_expert * prob_per_expert, dim=0)
        self.aux_loss = aux_loss

        output = torch.zeros_like(tokens)

        for expert_idx in range(self.n_experts):
            token_pos, slot_pos = torch.where(top_experts == expert_idx)

            selected_tokens = tokens[token_pos]
            expert_output = self.experts[expert_idx](selected_tokens)

            token_weights = top_weights[token_pos, slot_pos].unsqueeze(1)

            output.index_add_(dim=0, index=token_pos, source=expert_output * token_weights)

        return output.reshape(batch_size, seq_len, emb_dim)



class MultiQueryAttention(nn.Module):
    def __init__(self, cfg: dict[str, int | bool]):
        super().__init__()

        assert cfg["emb_dim"] % cfg["n_heads"] == 0

        self.emb_dim = cfg["emb_dim"]
        self.num_heads = cfg["n_heads"]
        self.head_dim = self.emb_dim // self.num_heads
        self.qkv_bias = cfg["qkv_bias"]

        self.query_proj = nn.Linear(self.emb_dim, self.emb_dim, self.qkv_bias)
        self.key_proj = nn.Linear(self.emb_dim, self.head_dim, self.qkv_bias)
        self.value_proj = nn.Linear(self.emb_dim, self.head_dim, self.qkv_bias)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        queries = self.query_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_proj(x).unsqueeze(1).expand(batch_size, self.num_heads, seq_len, self.head_dim)
        values = self.value_proj(x).unsqueeze(1).expand(batch_size, self.num_heads, seq_len, self.head_dim)

        context_vecs = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        context_vecs = context_vecs.transpose(1, 2).reshape(batch_size, seq_len, self.emb_dim)

        return self.out_proj(context_vecs)



class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attention = MultiQueryAttention(cfg)
        self.ff = MoE(cfg) if cfg["is_moe"] else FeedForward(cfg)
        self.norm1 = Norm(cfg)
        self.norm2 = Norm(cfg)

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x



class LLMConfig(PretrainedConfig):

    model_type = "custom_llm"

    def __init__(self,
                 vocab_size: int = 50257,
                 context_length: int = 1024,
                 emb_dim: int = 768,
                 hidden_dim: int = 3072,
                 n_heads: int = 12,
                 n_layers: int = 12,
                 qkv_bias: bool = False,
                 is_moe: bool = False,
                 n_experts: int | None = None,
                 top_k: int | None = None,
                 **kwargs):

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.qkv_bias = qkv_bias
        self.is_moe = is_moe
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.top_k = top_k

        super().__init__(**kwargs)

    def __getitem__(self, key):
        return getattr(self, key)



class LLM(PreTrainedModel):

    config_class = LLMConfig

    def __init__(self, cfg: LLMConfig):
        super().__init__(cfg)

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.trans_blocks = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg["n_layers"])]
        )

        self.norm = Norm(cfg)
        self.out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

        self.post_init()

    def forward(self, input_ids: torch.Tensor, **kwargs):
        _, seq_len = input_ids.shape
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=input_ids.device))

        logits = self.out(self.norm(self.trans_blocks(tok_emb + pos_emb)))
        return CausalLMOutputWithPast(logits=logits)