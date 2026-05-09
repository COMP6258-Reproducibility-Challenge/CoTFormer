"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import positional_encoders, caches

from .utils import LayerNorm


class InPlaceSetSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_val, dim):
        
        if last_slice is None:
            prev_length = 0
        else:
            prev_length = last_slice.shape[dim]
        new_length = prev_length + x_val.shape[dim]

        prefix_slice = [slice(None)] * dim 
        full_tensor[prefix_slice + [slice(prev_length, new_length)]] = x_val
        # print("writing cache slice:", prev_length, "to", new_length, "chunk size:", x_val.shape[dim]) #TODO: remove
        ctx.prev_length = prev_length
        ctx.new_length = new_length
        ctx.dim = dim
        ret = torch.Tensor().to(full_tensor)
        ret.set_(full_tensor[prefix_slice +[slice(None,new_length)]])
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        prefix_slice = [slice(None)] * ctx.dim 
        if ctx.prev_length == 0:
            return None, None, grad_out[prefix_slice + [slice(None, ctx.new_length)]], None #return one gradient for every input passed into forward excluding ctx
        else:
            return None, grad_out[prefix_slice + [slice(None, ctx.prev_length)]], grad_out[prefix_slice + [slice(ctx.prev_length, ctx.new_length)]], None
# second return value belongs to tokens generated in previous layers/repeats. Goes baclwards through last_slice. second belongs to token that was just generated now. Goes through x_val.

def apply_inplace_set(x_acc, x_val, dim):
    full_tensor, last_slice = x_acc
    new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_val, dim) # never call .forward() directly on torch.autograd.Function  
    return full_tensor, new_slice                                         # .apply(creates graph node generates ctx links unputs to engine then pytorch itself calls custom .forward(ctx...) running
                                                                           # running inplacesetslide.forward directly would mean it would run as standard python code and torch ignores during backward pass         

class CausalSelfAttention(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.cache_storage = lm_cache.get_storage_for_layer(self)
        self.config = config
        self.allow_cache_during_training = getattr(config, "allow_cache_during_training", False)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # print("Using flash attention:", self.flash)
        if self.flash:
            assert config.attention_window_length is None
        else:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = torch.tril(torch.ones(config.sequence_length, config.sequence_length)) # returns lower triangular
        if config.attention_window_length is not None:
            bias = torch.triu(bias, diagonal=-config.attention_window_length)
        self.register_buffer("bias", bias.view(1, 1, config.sequence_length, config.sequence_length))

        self.drop_cache()

    def init_cache(self, expected_total_length):
        self._lazy_init_cache_length = expected_total_length

    def drop_cache(self): # THIS is for the CoT cache not the KV cache.
        self.all_keys = None
        self.all_values = None
        self.all_indices = None
        self._lazy_init_cache_length = None
        

    def forward(self, x, pos_emb_closure, cache_context, start_index, indices): # TODO remove rep_idx and block_idx
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C = self.n_embd
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        pos_size = k.shape[-1] // 2

        q = pos_emb_closure.adapt_queries(q, start_index=start_index, indices=indices)
        if cache_context is not None and self.cache_storage is not None:
            att_prefix, cache_values_dict = \
                self.cache_storage.retrieve_for_query(q, cache_context, pos_emb_closure, start_index)
            if self.training and att_prefix is not None and not self.allow_cache_during_training:
                raise ValueError("Cache is not allowed during training")
        else:
            att_prefix = None
        k_before_pos = k
        k = pos_emb_closure.adapt_keys(k, start_index=start_index, indices=indices)

        if self._lazy_init_cache_length is not None:
            # assert indices is not None
            self.all_keys = (
                k.new_empty((B, self.n_head, self._lazy_init_cache_length, C // self.n_head)),
                None
            )
            self.all_values = (
                v.new_empty((B, self.n_head, self._lazy_init_cache_length, C // self.n_head)),
                None
            )
            self._lazy_init_cache_length = None
        
        if self.all_keys is not None:
            # assert indices is not None
            self.all_keys = apply_inplace_set(self.all_keys, k, dim=2)
            self.all_values = apply_inplace_set(self.all_values, v, dim=2)
            k = self.all_keys[1]
            v = self.all_values[1]
            # if not self.training:
                # print("q shape:", q.shape, "k shape:", k.shape, "v shape:", v.shape)
            attn_mask = self.bias[:,:,:T,:T].unsqueeze(3).repeat(
                1, 1, 1, k.shape[2] // T, 1
            ).unsqueeze(0).view(1, 1, q.shape[2], k.shape[2]) == 1
            is_causal = False
        else:
            attn_mask = None
            is_causal = True
        
        if self.flash and self.training: # in eval use normal attention so we can track metrics
            # print("WARNING: Using flash attention with PyTorch's built in scaled_dot_product_attention")
            if att_prefix is not None:
                raise NotImplementedError
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=is_causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = pos_emb_closure.adapt_attention_before_softmax(att, start_query_index=start_index, start_key_index=start_index)
            if attn_mask is None:
                attn_mask = self.bias[:,:,:T,:T] == 1
            att = att.masked_fill(~attn_mask, float('-inf'))
            if att_prefix is not None:
                prefix_size = att_prefix.shape[-1]
                current_size = att.shape[-1]
                att = torch.cat((att_prefix, att), dim=-1)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att) # this is what needs to be captured in order to see wether it attends to its own passed thoughts. maybe modify the function signature to accept output_att = True then append to the list that gets returned?
            if not self.training:                                 # new
                self.diagnose_attn = att.clone().detach() # NEW 
            if att_prefix is not None:
                att_prefix, att = torch.split(att, (prefix_size, current_size), dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            if att_prefix is not None:
                cache_v = cache_values_dict['v']
                if cache_v.ndim == v.ndim:
                    y += att_prefix @ cache_v
                elif cache_v.ndim == v.ndim + 1:
                    y += (att_prefix.unsqueeze(3) @ cache_v).squeeze(3)
                else:
                    raise NotImplementedError
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if cache_context is not None and self.cache_storage is not None:
            with torch.no_grad():
                self.cache_storage.store_in_cache(k_before_pos, {'v': v})
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, lm_cache)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, pos_emb_closure, cache_context, start_index, indices=None):
        x = x + self.attn(self.ln_1(x), pos_emb_closure, cache_context, start_index, indices)
        x = x + self.mlp(self.ln_2(x))
        return x
    # def forward(self, x, pos_emb_closure, cache_context, start_index, indices=None, rep_idx=None, block_idx=None):         # TODO remove
    #     x = x + self.attn(self.ln_1(x), pos_emb_closure, cache_context, start_index, indices, rep_idx=rep_idx, block_idx=block_idx)
    #     x = x + self.mlp(self.ln_2(x))
    #     return x

class LinearLearnedDepthPositionalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embs = nn.Embedding(1, config.n_embd)

    def forward(self, x, indices):
        emb = self.embs(torch.zeros_like(indices)) * ((indices).unsqueeze(-1))
        # emb = emb.view([1] * (len(x.shape) - 2) + list(emb.shape)).expand(*x.shape)
        return x + emb


class LearnedDepthPositionalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embs = nn.Embedding(config.n_repeat, config.n_embd)

    def forward(self, x, indices):
        emb = self.embs(indices)
        # emb = emb.view([1] * (len(x.shape) - 2) + list(emb.shape)).expand(*x.shape)
        return x + emb

class GPTBase(nn.Module):

    needs_iter = False

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.n_repeat = config.n_repeat

        if config.depth_embedding == "linear_learned":
            self.depth_emb = LinearLearnedDepthPositionalEncoder(config)
        elif config.depth_embedding == "learned":
            self.depth_emb = LearnedDepthPositionalEncoder(config)
        elif config.depth_embedding is None:
            self.depth_emb = None
        else:
            raise NotImplementedError(config.depth_embedding)
        self.lm_cache = caches.get_cache(config.lm_cache)(config)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = positional_encoders.get_encoder(config.positional_encoder)(config),
            drop = nn.Dropout(config.dropout),
            h_begin = nn.ModuleList( 
                [Block(config, self.lm_cache) for _ in range(config.n_layer_begin)]
            ),
            h_mid = nn.ModuleList(
                [Block(config, self.lm_cache) 
                for _ in range(config.n_layer_begin, config.n_layer - config.n_layer_end)],
            ),
            h_end = nn.ModuleList( 
                [Block(config, self.lm_cache) 
                for _ in range(config.n_layer - config.n_layer_end, config.n_layer)]),
            ln_mid = nn.Identity() if config.disable_ln_mid else LayerNorm(config.n_embd, bias=config.bias),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer)) # IMPORTANT: in our report lets make sure to talk about why projection layers are intialised like this.

        def _post_init_fn(module):
            if hasattr(module, "post_init"):
                module.post_init()
        self.apply(_post_init_fn)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.transformer.wpe.parameters()) # TODO: Why do we need this?
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, get_logits=False, use_cache=False, iter=None,log_metrics=False):
        device = idx.device
        b, t = idx.size()
        diag_metrics = {} # We will pack everything into a dict to keep it clean
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        if log_metrics and self.training:
            self.backward_metrics = {}
            self.forward_metrics = {}
        elif log_metrics and not self.training:
            self.forward_metrics = {}

        prev_x_last = None
        
        # forward the GPT model itself
        if use_cache:
            idx, index_shift, cache_context = self.lm_cache(idx)
        else:
            index_shift = 0
            cache_context = None
        if getattr(self.transformer.wpe, "needs_iter", False):
            idx, pos_emb_closure = self.transformer.wpe(idx, iter=iter) # position embeddings of shape (1, t, n_embd)
        else:
            idx, pos_emb_closure = self.transformer.wpe(idx) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(x)
        x = pos_emb_closure.adapt_model_input(x, start_index=index_shift)
       
        for block in self.transformer.h_begin:
            x = block(x, pos_emb_closure, cache_context, start_index=index_shift,) # NOTICE that they dont initialise the CoT cache for these blocks. KV cache is something entirely different than the CoT cache.
        
        B, T, D = x.shape

        total_expected_length = self.n_repeat * T
        for block in self.transformer.h_mid:                      # IMPORTANT: only initialise CoT cache for repeat blocks. EVERY SINGLE BLOCK has its own CoT cache and every single block can attend to past representations of that block only
            block.attn.init_cache(total_expected_length)
        if not self.training:
            x_into_mid = x.clone().detach() # We log the x going into mid blocks
            # diag_metrics['logit_lens'] = {}
            # lens_x_begin = self.transformer.ln_f(x_into_mid)
            # diag_metrics['logit_lens']['h_begin'] = self.lm_head(lens_x_begin).cpu()
            # Inside forward(), before the repeat loop:

            saved_hidden_states = {}
            saved_hidden_states['h_begin'] = x_into_mid
            num_mid_blocks = len(self.transformer.h_mid)
            halfway_idx = num_mid_blocks // 2
        sum_active = 0
        # for rep_idx in range(1, self.n_repeat+1):
            
        #     for mid_idx, block in enumerate(self.transformer.h_mid):
        #         x = block(x, pos_emb_closure, cache_context, start_index=index_shift,
        #         rep_idx = rep_idx,
        #         block_idx = mid_idx) 
        if not hasattr(self, 'backward_metrics'):
            self.backward_metrics = {}
        for rep_idx in range(1, self.n_repeat+1):
            if self.depth_emb is not None:
                x = self.depth_emb(x, indices=x.new_full((B, T), self.n_repeat - rep_idx, dtype=torch.long))

            for block_idx, block in enumerate(self.transformer.h_mid):
                x = block(x, pos_emb_closure, cache_context, start_index=index_shift)        # for logit lens or something similar take the current hidden state x, pass it through the final layer norm (i assume self.transformer.ln_f(x)) and shove that through the lm head prematurely (we do this to empirically measure representations getting more mature or ritcher or whatever you wanna call it)
                if log_metrics and not self.training:
                    # Capture halfway through the repeat (e.g., 1.5)
                    if block_idx == halfway_idx - 1:
                        saved_hidden_states[f'rep_{rep_idx}.5'] = x.clone().detach()
                    # Capture at the end of the repeat (e.g., 1.0)
                    elif block_idx == num_mid_blocks - 1:
                        saved_hidden_states[f'rep_{rep_idx}.0'] = x.clone().detach()
            if log_metrics or not self.training:
                x_outof_mid = x.clone().detach()
                curr_x_last = x[:, -1, :].clone().detach() # we take the last token in the sequence 
                # pre ln var
                var_val = curr_x_last.var(dim=-1).mean().item()
                self.forward_metrics[f'var_end_rep_{rep_idx}'] = var_val
                if prev_x_last is not None:
                        # Calculate Cosine Similarity across the hidden dimension
                        cos_sim = F.cosine_similarity(curr_x_last, prev_x_last, dim=-1).mean().item()
                        cos_dist = 1.0 -cos_sim
                        self.forward_metrics[f'cos_dist_{rep_idx}_vs_{rep_idx-1}'] = cos_dist


                prev_x_last = curr_x_last
                # if not self.training:
                #     # We need the whole sequence [B, T, D] for the heatmaps, so we don't slice [:, -1, :]
                #     lens_x_rep = self.transformer.ln_f(x.clone().detach())
                #     diag_metrics['logit_lens'][f'rep_{rep_idx}'] = self.lm_head(lens_x_rep).cpu()

            if self.training and x.requires_grad and log_metrics:
                def make_hook(current_rep):
                    def hook(grad):
                        # Norm of the gradient for the last token
                        grad_norm = grad[:, -1, :].norm(p=2).item()
                        self.backward_metrics[f'grad_norm_enter_rep_{current_rep}'] = grad_norm
                    return hook
                
                x.register_hook(make_hook(rep_idx))
            x = self.transformer.ln_mid(x) # this is the output of the repeat blocks, right before going into the final blocks. we will use this representation to track metrics about how the representations evolve across the repeat blocks. we will also use it for the logit lens in the report.    

        if not self.training:
            # x_outof_mid = x.clone().detach() 
            sim_of_xs = F.cosine_similarity(x_into_mid, x_outof_mid, dim=-1).mean().item()
            var_into = x_into_mid.var(dim=-1).mean().item()
            var_outof = x_outof_mid.var(dim=-1).mean().item()
            

            end_att = self.transformer.h_mid[-1].attn.diagnose_attn # end_att shape (B, H, Q, K_tot):  torch.Size([2, 12, 256, 1280])bathc size head dim query total k
            B_d, H_d, Q_d, K_tot = end_att.shape 
            
            # 1. Base Reshape: (B, H, Q, R, Q)
            reshaped_att = end_att.view(B_d, H_d, Q_d, self.n_repeat, Q_d) #convert from (B, H, Q, K_tot) to (B, H, Q, R, Q) where R is the repeat dimension and the last Q is the token dimension inside each repeat. K_tot should equal R * Q.
                # split the full key axis into 5 chunks, one chunk per repeat, each chunk containing the 256 token positions from that repeat
            # 2. Repeat Probabilitiy mass: (B, H, Q, R)
            repeat_mass = reshaped_att.sum(dim=-1) 

            # 3. Repeat Entropy: (B, H, Q)
            repeat_entropy = -(repeat_mass * torch.log(repeat_mass + 1e-9)).sum(dim=-1) # B, H Q, R -> B, H, Q  this is too big to make sense out of so we will use statistics on it later

            # 4. Within-Repeat Entropy: (B, H, Q, R)
            local_p = reshaped_att / (repeat_mass.unsqueeze(-1) + 1e-9) # unsqueeze adds fake dimension for broadcasting   #  B, h q r q / (B, h q r 1) -> (B, H, Q, R, Q)  this is the local distribution of attention within each repeat. how is the attention distributed across the tokens inside each repeat? if it attends to one token a lot then low entropy if it attends to many tokens more evenly then high entropy    
            within_repeat_entropy = -(local_p * torch.log(local_p + 1e-9)).sum(dim=-1) # (B, H, Q, R, Q) -> (B, H, Q, R) this is the within repeat entropy. for each repeat, how focused or smeared is the attention across the tokens in that repeat? if it attends to one token a lot then low entropy if it attends to many tokens more evenly then high entropy

            # 5. Same-Position vs Different-Position Budget: (B, H, Q, R) so you extract the diagonal between the two Q dimensions (dim 2 and dim 4)
            # torch.diagonal moves the new diagonal dim to the very end -> (B, H, R, Q)
            # you transpose it back to match the pmass shape -> (B, H, Q, R)
            same_pos = torch.diagonal(reshaped_att, dim1=2, dim2=4).transpose(-1, -2) # reason for the transpose: https://docs.pytorch.org/docs/stable/generated/torch.diagonal.html
            different_pos = repeat_mass - same_pos 

            '''
            NOTE
            so the initial endatt is of size batch head q ktotal then you reshape it into self.n_repeat,
            q_d (last 2 dims) so that you get the per repeat attentions then you sum along the last q so you
            get per repeat probability mass then you calculate inter repeat entropy by which is a b h q because you summed along 
            R then to calculate within repeat entropy with unsqueeze you add a dummy dimension and since this is not matmul but mat divide 
            it just divides each repeat by that repeat so it normalises to 1 then you calculate within repeat entropy from that then you check 
            same position vs different position budget to do that you extract diagonal between the two Q dimensions dim 2 and dim 4 torch.diag appends 
            it to the end so transpose it to get how much it attended to the same pos 
            then just subtract from the repeat mass to get the difference also reshape 
            '''
                        # 2. Repeat Probabilitiy mass: (B, H, Q, R)
                                    # 3. Repeat Entropy: (B, H, Q)
                                        # 4. Within-Repeat Entropy: (B, H, Q, R)



            diag_metrics['macro_budget'] = repeat_mass.mean(dim=(0, 1, 2)).cpu().numpy() # (R,) ACCRoss the entire network what was the average percentages of attention?
            diag_metrics['macro_rep_entropy'] = repeat_entropy.mean().item()        # Scalar        # overall is the model's attention focused on specific looops or is it semared
            diag_metrics['macro_in_entropy'] = within_repeat_entropy.mean(dim=(0, 1, 2)).cpu().numpy() # (R,)     in a certain loop does it focus on  a specific token or smear?
            diag_metrics['macro_same_pos'] = same_pos.mean(dim=(0, 1, 2)).cpu().numpy() # (R,)    # on average how much of the attention budget in a loop was spent looking at the exact same position
            
            # Per-head stats (keeping the H dimension)
            diag_metrics['head_budget'] = repeat_mass.mean(dim=(0, 2)).cpu().numpy()     # (H, R)
            diag_metrics['head_rep_entropy'] = repeat_entropy.mean(dim=(0, 2)).cpu().numpy() # (H,)
            
            # --- APPENDIX F HEATMAP DATA (Store for later) ---
            # "attention from the last token in sequence to all other token-repeat pairs"
            # We grab the last query token (Q_d - 1) and average across the batch.
            # Shape becomes: (H, R, Q)
            diag_metrics['appendix_f_heatmap'] = reshaped_att[:, :, -1, :, :].mean(dim=0).cpu().numpy()
            '''
            NOTE
            THIS HEATMAP WILL BE USED TO GENERATE VIS FOR APPANDIX F
            import seaborn as sns
            import matplotlib.pyplot as plt

            # head_data shape is (5, 256) -> 5 repeats, 256 tokens
            head_data = appendix_f_heatmap[0] # Select Head 0
            sns.heatmap(head_data, cmap="viridis")
            plt.xlabel("Token Index")
            plt.ylabel("Repeat Index")
            plt.show()
            '''
        for block in self.transformer.h_mid: # When n repeat blocks are finished, we drop the CoT cache.
            block.attn.drop_cache() # NOTICE again this is not the same thing as kv cache in the common sense


        for block in self.transformer.h_end:
            x = block(x, pos_emb_closure, cache_context, start_index=index_shift)
            

        if not self.training:
                saved_hidden_states[f'final'] = x.clone().detach()

            
            
        
        x = self.transformer.ln_f(x)

        if use_cache:
            x = self.lm_cache.get_final_logits(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            cross_entropy_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = cross_entropy_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            cross_entropy_loss = None
        logits = logits if get_logits else None
        if log_metrics and not self.training:
            diag_metrics['hidden_states'] = saved_hidden_states

        return {'logits': logits, 
                'loss': loss, 
                'cross_entropy_loss': cross_entropy_loss,
                'average_depth': torch.as_tensor(self.n_repeat) * len(self.transformer.h_mid) + len(self.transformer.h_begin) + len(self.transformer.h_end),
                'sim_of_xs': sim_of_xs if not self.training else None,   # NEW
                'var_into': var_into if not self.training else None,       #NEW
                'var_outof': var_outof if not self.training else None,          #NEW 
                'diag_metrics': diag_metrics if not self.training else None          #NEW 
                }
    def clear_state(self):
        self.lm_cache.clear_state() # NOTICE that this clears the KV cache. and NOT the CoT cache.

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
 