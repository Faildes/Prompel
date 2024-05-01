import os
import re
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import hashlib
from io import BytesIO
import safetensors.torch
from typing import Callable, Union, Optional


re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    return key

def safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def lbw_lora(input_, output, ratios):
    print("Apply LBW")

    assert isinstance(input_, str)
    assert isinstance(output, str)
    assert isinstance(ratios, str)
    assert os.path.exists(input_), f"{input_} is not exists"
    assert os.path.exists(output) == False, f"{output} aleady exists"

    LOAD_PATH = input_
    SAVE_PATH = output
    RATIOS = [float(x) for x in ratios.split(",")]
    LAYERS = len(RATIOS)
    assert LAYERS in [17, 26]

    BLOCKID17 = [
        "BASE", "IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "M00",
        "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]
    BLOCKID26 = [
        "BASE", "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08", "IN09", "IN10", "IN11", "M00",
        "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]

    if LAYERS == 17:
        RATIO_OF_ = dict(zip(BLOCKID17, RATIOS))
    if LAYERS == 26:
        RATIO_OF_ = dict(zip(BLOCKID26, RATIOS))
    print(RATIO_OF_)

    PATTERNS = [
        r"^transformer_text_model_(encoder)_layers_(\d+)_.*",
        r"^diffusion_model_(in)put_blocks_(\d+)_.*",
        r"^diffusion_model_(middle)_block_(\d+)_.*",
        r"^diffusion_model_(out)put_blocks_(\d+)_.*"]

    def replacement(match):
        g1 = str(match.group(1))  # encoder, in, middle, out
        g2 = int(match.group(2))  # number
        assert g1 in ["encoder", "in", "middle", "out"]
        assert isinstance(g2, int)

        if g1 == "encoder":
            return "BASE"
        if g1 == "middle":
            return "M00"
        return f"{str.upper(g1)}{g2:02}"

    def compvis_name_to_blockid(compvis_name):
        strings = compvis_name
        for pattern in PATTERNS:
            strings = re.sub(pattern, replacement, strings)
            if strings != compvis_name:
                break
        assert strings != compvis_name
        blockid = strings

        if LAYERS == 17:
            assert blockid in BLOCKID26, f"Incorrect layer {blockid}"
            assert blockid in BLOCKID17, f"{blockid} is not included in 17 layers. May be 26 layers?"
        if LAYERS == 26:
            assert blockid in BLOCKID26, f"Incorrect layer {blockid}" 
        return blockid

    with safe_open(LOAD_PATH, framework="pt", device="cpu") as f:
        tensors = {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key)  # key = diffusers_name
            compvis_name = convert_diffusers_name_to_compvis(key, is_sd2=False)
            blockid = compvis_name_to_blockid(compvis_name)
            if compvis_name.endswith("lora_up.weight"):
                tensors[key] *= RATIO_OF_[blockid]
                print(f"({blockid}) {compvis_name} "
                      f"updated with factor {RATIO_OF_[blockid]}")
        
        save_file(tensors, SAVE_PATH)

    print("Done")
