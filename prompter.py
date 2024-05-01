import torch
import math
import re
import os
from safetensors.torch import load_file
from .block_lora import lbw_lora
from typing import Union, Optional, List, Tuple

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

class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)
        
def get_multicond_prompt_list(prompt: str):
    res_indexes = []

    prompt_indexes = {}
    subprompts = re_AND.split(prompt)

    indexes = []
    for subprompt in subprompts:
        match = re_weight.search(subprompt)

        text, weight = match.groups() if match is not None else (subprompt, 1.0)

        weight = float(weight) if weight is not None else 1.0

        index = prompt_indexes.get(text, None)
        if index is None:
            index = len(prompt_flat_list)
            prompt_flat_list.append(text)
            prompt_indexes[text] = index

        indexes.append((index, weight))

    res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes

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


class CLIPTextCustomEmbedder(object):
    def __init__(self, tokenizer, text_encoder, device,
                 clip_stop_at_last_layers=1):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.token_mults = {}
        self.device = device
        self.clip_stop_at_last_layers = clip_stop_at_last_layers

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
    
    def get_token_ids(self, texts):
        if type(texts) == str:
            texts=[texts]
        token_ids_list = self.tokenizer(
            texts,
            truncation=False,
            return_tensors=None)['input_ids']
        return token_ids_list

    def get_pooled(self, texts):
        device = self.device
        token_ids = self.get_token_ids(texts)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

        text_encoder_output = self.text_encoder(token_ids, None, return_dict=True)
        pooled = text_encoder_output.text_embeds

        return pooled
        
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


def text_embeddings_equal_len(text_embedder, prompt, negative_prompt):
    cond_embeddings = text_embedder(prompt)
    uncond_embeddings = text_embedder(negative_prompt)

    cond_len = cond_embeddings.shape[1]
    uncond_len = uncond_embeddings.shape[1]
    if cond_len == uncond_len:
        return [cond_embeddings, uncond_embeddings]
    else:
        if cond_len > uncond_len:
            n = (cond_len - uncond_len) // 77
            return [cond_embeddings, torch.cat([uncond_embeddings] + [text_embedder("")]*n, dim=1)]
        else:
            n = (uncond_len - cond_len) // 77
            return [torch.cat([cond_embeddings] + [text_embedder("")]*n, dim=1), uncond_embeddings]
        
def text_pooled(text_embedder, positive, negative):
    cond_embeddings = text_embedder.get_pooled(positve)
    uncond_embeddings = text_embedder.get_pooled(negative)
    cond_len = cond_embeddings.shape[1]
    uncond_len = uncond_embeddings.shape[1]
    if cond_len == uncond_len:
        return [cond_embeddings, uncond_embeddings]
    else:
        if cond_len > uncond_len:
            n = (cond_len - uncond_len) // 77
            return [cond_embeddings, torch.cat([uncond_embeddings] + [text_embedder("")]*n, dim=1)]
        else:
            n = (uncond_len - cond_len) // 77
            return [torch.cat([cond_embeddings] + [text_embedder("")]*n, dim=1), uncond_embeddings]
        
def text_embeddings(pipe, prompt, negative_prompt, clip_stop_at_last_layers, require_pooled):
    text_embedder = CLIPTextCustomEmbedder(tokenizer=pipe.tokenizer,
                                           text_encoder=pipe.text_encoder,
                                           device=pipe.text_encoder.device,
                                           clip_stop_at_last_layers=clip_stop_at_last_layers)
    res = text_embeddings_equal_len(text_embedder, prompt, negative_prompt)
    nk = {"prompt_embeds":res[0],
          "negative_prompt_embeds":res[1]}
    if require_pooled:
        resp = text_pooled(text_embedder, prompt, negative_prompt)
        nk = {"prompt_embeds":res[0],
              "pooled_prompt_embeds":resp[0],
              "negative_prompt_embeds":res[1],
              "negative_pooled_prompt_embeds":resp[1]}
    return nk

def apply_embeddings(pipe, input_str,epath,isxl=False):
    if isxl:
        for name, path in epath.items():
            new_string = input_str.replace(name,"")

            if new_string != input_str:
                state = load_file(path)
                pipe.load_textual_inversion(pretrained_model_name_or_path=state["clip_g"], token=name, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2, local_files_only=True)
                pipe.load_textual_inversion(pretrained_model_name_or_path=state["clip_l"], token=name, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, local_files_only=True)
    else:
        for name, path in epath.items():
            new_string = input_str.replace(name,"")

            if new_string != input_str:
                pipe.load_textual_inversion(pretrained_model_name_or_path=path, token=name, local_files_only=True)

    return input_str

def lora_prompt(prompt, pipe, lpath):
    loras = []
    adap_list=[]
    alphas=[]
    add = []
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
    
def create_conditioning(pipe, positive: str, negative: str, epath, lora_list, clip_skip: int = 1, require_pooled: Union[bool, List[bool]] = False):
    positive = apply_embeddings(pipe, positive, epath, isxl=(require_pooled != False))
    negative = apply_embeddings(pipe, negative, epath, isxl=(require_pooled != False))
 
    positive, lora_list = lora_prompt(positive, pipe, lora_list)
    embeds = text_embeddings(pipe, positive, negative, clip_skip, require_pooled)
    return (embeds, lora_list)
