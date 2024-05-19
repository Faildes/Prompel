import torch
import math
import re
import os
from .block_lora import lbw_lora
from typing import Union, Optional, List, Tuple
from abc import ABC

class BaseTextualInversionManager(ABC):
    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: List[int]) -> List[int]:
        raise NotImplementedError()

class DiffusersTextualInversionManager(BaseTextualInversionManager):
    """
    A textual inversion manager for use with diffusers.
    """
    def __init__(self, pipe):
        self.pipe = pipe

    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) == 0:
            return token_ids

        prompt = self.pipe.tokenizer.decode(token_ids)
        prompt = self.pipe.maybe_convert_prompt(prompt, self.pipe.tokenizer)
        return self.pipe.tokenizer.encode(prompt, add_special_tokens=False)


re_attention = re.compile(r"""
\\\(|
\\\{|
\\\)|
\\\}|
\\\[|
\\]|
\\\\|
\\|
\(|
\{|
\[|
:([+-]?[.\d]+)\)|
\)|
\}|
]|
[^\\()\\{}\[\]:]+|
:
""", re.X)
re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")
re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its assoicated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

class CLIPMultiTextCustomEmbedder(object):
    def __init__(self, tokenizers, text_encoders, device, textual_inversion_manager: BaseTextualInversionManager,
                 clip_stop_at_last_layers=1, requires_pooled=False):
        self.tokenizer = tokenizers[0]
        self.tokenizer_2 = tokenizers[1]
        self.text_encoder = text_encoders[0]
        self.text_encoder_2 = text_encoders[1]
        self.textual_inversion_manager = textual_inversion_manager
        self.token_mults = {}
        self.device = device
        self.clip_stop_at_last_layers = clip_stop_at_last_layers
        self.requires_pooled=requires_pooled

    def tokenize_line(self, line):
        def get_target_prompt_token_count(token_count):
            return math.ceil(max(token_count, 1) / 75) * 75
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
        parsed = parse_prompt_attention(line)
        tokenized = self.tokenizer(
            [text for text, _ in parsed], truncation=False,
            add_special_tokens=False)["input_ids"]

        fixes = []
        remade_tokens = []
        multipliers = []
        
        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]
                remade_tokens.append(token)
                multipliers.append(weight)
                i += 1

        token_count = len(remade_tokens)
        prompt_target_length = get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)
        remade_tokens = remade_tokens + [id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add
        return remade_tokens, fixes, multipliers, token_count

    def process_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        remade_batch_tokens = []
        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, multipliers = cache[line]
            else:
                remade_tokens, _, multipliers, _ = self.tokenize_line(line)
                cache[line] = (remade_tokens, multipliers)
            remade_batch_tokens.append(remade_tokens)
            batch_multipliers.append(multipliers)

        return batch_multipliers, remade_batch_tokens

    def __call__(self, text):
        batch_multipliers, remade_batch_tokens = self.process_text(text)

        z = None
        pooled = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]
            rem_multipliers = [x[75:] for x in batch_multipliers]

            tokens = []
            multipliers = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:75])
                    multipliers.append(batch_multipliers[j][:75])
                else:
                    tokens.append([self.tokenizer.eos_token_id] * 75)
                    multipliers.append([1.0] * 75)

            z1, pooly = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), axis=-2)
            pooled = pooly if pooled is None else torch.cat((pooled, pooly), axis=-2)

            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1

        return z, pooled

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        remade_batch_tokens = [[self.tokenizer.bos_token_id] + x[:75] +
                               [self.tokenizer.eos_token_id] for x in remade_batch_tokens]
        batch_multipliers = [[1.0] + x[:75] + [1.0] for x in batch_multipliers]

        tokens = torch.asarray(remade_batch_tokens)
        # print(tokens.shape)
        # print(tokens)
        encoder=[self.text_encoder,self.text_encoder_2]
        plist=[]
        batch_multipliers_of_same_length = [
            x + [1.0] * (75 - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(
            batch_multipliers_of_same_length).to(self.device)
        for text_encoder in encoder:
            output = text_encoder(tokens.to(self.device),output_hidden_states=True, return_dict=False)
            pooled=output[0]
            if self.clip_stop_at_last_layers > 1:
                z = output[-1][-(2+self.clip_stop_at_last_layers)]
            else:
                z = output[-1][-2]
            bs_embed, seq_len, _ = z.shape
            z = z.view(bs_embed, seq_len, -1)
            z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
            plist.append(z)
        
        
        z = torch.concat(plist, dim=-1)
        pooled = pooled.view(bs_embed, -1)
        return z, pooled

    def get_text_tokens(self, text):
        batch_multipliers, remade_batch_tokens = self.process_text(text)
        return [[self.tokenizer.bos_token_id] + remade_batch_tokens[0]], \
            [[1.0] + batch_multipliers[0]]

    def get_token_ids(self, texts: List[str]) -> List[List[int]]:
        token_ids_list = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            return_tensors=None,  # just give me lists of ints
        )['input_ids']

        result = []
        for token_ids in token_ids_list:
            # trim eos/bos
            token_ids = token_ids[1:-1]
            # pad for textual inversions with vector length >1
            if self.textual_inversion_manager is not None:
                token_ids = self.textual_inversion_manager.expand_textual_inversion_token_ids_if_necessary(token_ids)

            token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]

            result.append(token_ids)

        return result

    def get_pooled_embeddings(self, texts: List[str], attention_mask: Optional[torch.Tensor]=None, device: Optional[str]=None) -> Optional[torch.Tensor]:
        
        device = device or self.device

        token_ids = self.get_token_ids(texts)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

        text_encoder_output = self.text_encoder(token_ids, attention_mask, return_dict=True)
        pooled = [text_encoder_output.pooler_output]
        pooled = torch.cat(pooled, dim=-1)

        return pooled

class CLIPTextCustomEmbedder(object):
    def __init__(self, tokenizer, text_encoder, device, textual_inversion_manager: BaseTextualInversionManager,
                 clip_stop_at_last_layers=1, requires_pooled=False):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.textual_inversion_manager = textual_inversion_manager
        self.token_mults = {}
        self.device = device
        self.clip_stop_at_last_layers = clip_stop_at_last_layers
        self.requires_pooled=requires_pooled

    def tokenize_line(self, line):
        def get_target_prompt_token_count(token_count):
            return math.ceil(max(token_count, 1) / 75) * 75

        id_end = self.tokenizer.eos_token_id
        parsed = parse_prompt_attention(line)
        tokenized = self.tokenizer(
            [text for text, _ in parsed], truncation=False,
            add_special_tokens=False)["input_ids"]

        fixes = []
        remade_tokens = []
        multipliers = []

        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]
                remade_tokens.append(token)
                multipliers.append(weight)
                i += 1

        token_count = len(remade_tokens)
        prompt_target_length = get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)
        remade_tokens = remade_tokens + [id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add
        return remade_tokens, fixes, multipliers, token_count

    def process_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        remade_batch_tokens = []
        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, fixes, multipliers = cache[line]
            else:
                remade_tokens, fixes, multipliers, _ = self.tokenize_line(line)
                cache[line] = (remade_tokens, fixes, multipliers)

            remade_batch_tokens.append(remade_tokens)
            batch_multipliers.append(multipliers)

        return batch_multipliers, remade_batch_tokens

    def __call__(self, text):
        batch_multipliers, remade_batch_tokens = self.process_text(text)

        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]
            rem_multipliers = [x[75:] for x in batch_multipliers]

            tokens = []
            multipliers = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:75])
                    multipliers.append(batch_multipliers[j][:75])
                else:
                    tokens.append([self.tokenizer.eos_token_id] * 75)
                    multipliers.append([1.0] * 75)

            z1 = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), axis=-2)

            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1

        return z

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        remade_batch_tokens = [[self.tokenizer.bos_token_id] + x[:75] +
                               [self.tokenizer.eos_token_id] for x in remade_batch_tokens]
        batch_multipliers = [[1.0] + x[:75] + [1.0] for x in batch_multipliers]

        tokens = torch.asarray(remade_batch_tokens).to(self.device)
        # print(tokens.shape)
        # print(tokens)
        outputs = self.text_encoder(
            input_ids=tokens, output_hidden_states=True)

        if self.clip_stop_at_last_layers > 1:
            z = self.text_encoder.text_model.final_layer_norm(
                outputs.hidden_states[-self.clip_stop_at_last_layers])
        else:
            z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well
        # to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [
            x + [1.0] * (75 - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(
            batch_multipliers_of_same_length).to(self.device)
        # print(batch_multipliers.shape)
        # print(batch_multipliers)

        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape +
                                       (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z

    def get_text_tokens(self, text):
        batch_multipliers, remade_batch_tokens = self.process_text(text)
        return [[self.tokenizer.bos_token_id] + remade_batch_tokens[0]], \
            [[1.0] + batch_multipliers[0]]

    def get_token_ids(self, texts: List[str]) -> List[List[int]]:
        token_ids_list = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            return_tensors=None,  # just give me lists of ints
        )['input_ids']

        result = []
        for token_ids in token_ids_list:
            # trim eos/bos
            token_ids = token_ids[1:-1]
            # pad for textual inversions with vector length >1
            if self.textual_inversion_manager is not None:
                token_ids = self.textual_inversion_manager.expand_textual_inversion_token_ids_if_necessary(token_ids)

            # add back eos/bos if requested
            if include_start_and_end_markers:
                token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]

            result.append(token_ids)

        return result

    def get_pooled_embeddings(self, texts: List[str], attention_mask: Optional[torch.Tensor]=None, device: Optional[str]=None) -> Optional[torch.Tensor]:
        
        device = device or self.device

        token_ids = self.get_token_ids(texts)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

        text_encoder_output = self.text_encoder(token_ids, attention_mask, return_dict=True)
        pooled = text_encoder_output.pooler_output

        return pooled

def build_conditioning_tensor(text_embedder, text: str) -> torch.Tensor:
    conditioning = text_embedder(text)
    if text_embedder.requires_pooled:
        return conditioning
    else:
        return [conditioning]

def text_embeddings_equal_len(text_embedder, prompt, negative_prompt) -> List[torch.Tensor]:
    conds = build_conditioning_tensor(text_embedder, prompt)
    unconds = build_conditioning_tensor(text_embedder, negative_prompt)
    if len(conds)==1:
        conditionings = [conds,unconds]
        pooled = None
    else:
        conditionings = [conds[0], unconds[0]]
        pooled = [conds[1], unconds[1]]
    
    cond_len = conditionings[0].shape[1]
    uncond_len = conditionings[1].shape[1]
    if cond_len == uncond_len:
        return conditionings, pooled
    else:
        if cond_len > uncond_len:
            n = (cond_len - uncond_len) // 77
            return [conditionings[0], torch.cat([conditionings[1]] + [text_embedder("")]*n, dim=1)],pooled
        else:
            n = (uncond_len - cond_len) // 77
            return [torch.cat([conditionings[0]] + [text_embedder("")]*n, dim=1), conditionings[1]],pooled

def text_embeddings(pipe, prompt, negative_prompt, clip_stop_at_last_layers=1, pool=False):
    if pool:
        text_embedder = CLIPMultiTextCustomEmbedder(tokenizers=[pipe.tokenizer,pipe.tokenizer_2],
                                                   text_encoders=[pipe.text_encoder,pipe.text_encoder_2],
                                                   device=pipe.text_encoder.device,
                                                   textual_inversion_manager=DiffusersTextualInversionManager(pipe),
                                                   clip_stop_at_last_layers=clip_stop_at_last_layers,
                                                   requires_pooled=pool)
    else:
        text_embedder = CLIPTextCustomEmbedder(tokenizer=pipe.tokenizer,
                                               text_encoder=pipe.text_encoder,
                                               device=pipe.text_encoder.device,
                                               textual_inversion_manager=DiffusersTextualInversionManager(pipe),
                                               clip_stop_at_last_layers=clip_stop_at_last_layers,
                                               requires_pooled=pool)
    conds, pooled = text_embeddings_equal_len(text_embedder, prompt, negative_prompt)
    return conds, pooled

def apply_embeddings(pipe, input_str,epath,pool):
    for name, path in epath.items():
        new_string = input_str.replace(name,"")

        if new_string != input_str:
            if pool:
                from safetensors.torch import load_file
                state = load_file(path)
                pipe.load_textual_inversion(pretrained_model_name_or_path=state["clip_g"], token=name, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2, local_files_only=True)
                pipe.load_textual_inversion(pretrained_model_name_or_path=state["clip_l"], token=name, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, local_files_only=True)
            else:
                pipe.load_textual_inversion(pretrained_model_name_or_path=path, token=name, local_files_only=True)

    return input_str

def bpro(prompt):
    k = prompt.split(",")
    thu = []
    for g in k:
        f = g.count(" ")
        thu.append([g, f+1])
    off = 0
    nl = []
    t = 0
    for x in thu:
        if "BREAK" in x[0]:
            tok = t+off
            add = tok % 75
            nl += [" "]*add
            off += add
            continue
        t += x[1]
        nl.append(x[0])
    return ",".join(nl)

def lora_prompt(prompt, pipe, lpath):
    loras = []
    adap_list=[]
    alphas=[]
    add = []
    prompt=bpro(prompt)
    def network_replacement(m):
        alias = m.group(1)
        num = m.group(2)
        try:
            data = lpath[alias]
            mpath = data[1]
            dpath = data[0]
            alias = data[2]
        except:
            return ""
        if "|" in num:
            t = num.split("|")
            alpha = float(t[0])
            apply = t[1]
            npath = f"{mpath}{alias}_{apply}.safetensors"
            try:
              data = lpath[f"{alias}_{apply}"]
              loras.append([data[0], alpha])
              return alias
            except:
              lpath[f"{alias}_{apply}"] = [npath, dpath]
              lbw_lora(dpath, npath, apply)
              dpath = npath
        else:
            alpha = float(num)
        loras.append([dpath, alpha])
        return alias
    re_lora = re.compile("<lora:([^:]+):([^:]+)>")
    prompt = re.sub(re_lora, network_replacement, prompt)
    for k in loras:
        p = os.path.abspath(os.path.join(k[0], ".."))
        safe = os.path.basename(k[0])
        name = os.path.splitext(safe)[0].replace(".","_")
        alphas.append(k[1])
        adap_list.append(name)
        try:
          pipe.load_lora_weights(p, weight_name=safe, adapter_name=name)
        except:
          pass
    pipe.set_adapters(adap_list, adapter_weights=alphas)
    return prompt, lpath
    
def create_conditioning(pipe, positive: str, negative: str, epath, lora_list, clip_skip = 1, pool = False):
    positive = apply_embeddings(pipe, positive, epath, pool)
    negative = apply_embeddings(pipe, negative, epath, pool)
 
    positive, lora_list = lora_prompt(positive, pipe, lora_list)

    conds, pooled = text_embeddings(pipe, positive, negative, clip_skip, pool)
    if pool:
        embed_dict={
            'prompt_embeds': conds[0],
            'pooled_prompt_embeds':pooled[0],
            'negative_prompt_embeds': conds[1],
            'negative_pooled_prompt_embeds': pooled[1],
        }

    else:
        embed_dict={
            'prompt_embeds': conds[0],
            'negative_prompt_embeds': conds[1],
        }
    
    return (embed_dict, lora_list)
