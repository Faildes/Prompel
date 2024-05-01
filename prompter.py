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

class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []

class CLIPTextCustomEmbedder(object):
    def __init__(self, tokenizer, text_encoder, device,
                 clip_stop_at_last_layers=1):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.token_mults = {}
        self.device = device
        self.chunk_length = 75
        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.id_end
        self.clip_stop_at_last_layers = clip_stop_at_last_layers

    def empty_chunk(self):
        """creates an empty PromptChunk and returns it"""

        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def tokenize_line(self, line):
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        """

        parsed = parse_prompt_attention(line)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                # this is when we are at the end of allotted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                elif opts.comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= 20:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count
    
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
        remade_batch_tokens = token_ids
        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]

            tokens = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:75])
                else:
                    tokens.append([self.tokenizer.eos_token_id] * 75)
            rbt = [[self.tokenizer.bos_token_id] + x[:75] +
                   [self.tokenizer.eos_token_id] for x in remade_batch_tokens]
            tokens = torch.asarray(rbt).to(self.device)
            outputs = self.text_encoder(
                input_ids=tokens, output_hidden_states=True)

            if self.clip_stop_at_last_layers > 1:
                z1 = self.text_encoder.text_model.final_layer_norm(
                    outputs.hidden_states[-self.clip_stop_at_last_layers])
            else:
                z1 = outputs.last_hidden_state

            z = z1 if z is None else torch.cat((z, z1), axis=-2)

            remade_batch_tokens = rem_tokens
            i += 1

        return z
        
    def process_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        token_count = 0
        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)
                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def __call__(self, text, pool=False):
        batch_chunks, token_count = self.process_text(text)
        z = None
        i = 0
        zs = []
        chunk_count = max([len(x) for x in batch_chunks])
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]
            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunks]

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        if pool:
            return torch.hstack(zs), zs[0].pooled
        return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
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
        pooled = getattr(z, "pooled", None)
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
        if pooled is not None:
            z.pooled = pooled
            
        return z

    def get_text_tokens(self, text):
        batch_multipliers, remade_batch_tokens = self.process_text(text)
        return [[self.tokenizer.bos_token_id] + remade_batch_tokens[0]], \
            [[1.0] + batch_multipliers[0]]


def text_embeddings_equal_len(text_embedder, prompt, negative_prompt, pool):
    cond_embeddings = text_embedder(prompt, pool)
    uncond_embeddings = text_embedder(negative_prompt, pool)

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
    res = text_embeddings_equal_len(text_embedder, prompt, negative_prompt, pool=require_pooled)
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
        from safetensors.torch import load_file
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
