import os
import json
import struct

import safetensors
import torch

import argparse

"""
Usage:
  python export_mistral.py --model_dir /path/to/Mistral-7B-v0.1 [--out model.bin] [--quant int8]

Arguments:
  --model_dir   Required. Path to the Hugging Face model directory.
  --out         Optional. Output file path. Defaults to ./model.bin
  --quant       Optional. Quantization mode. Defaults to f32. Accepts f32 or int8
  

Examples:
  # Full precision (float32)
  python export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ./mistral-f32.bin --quant f32

  # 8-bit quantization
  python export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ./mistral-int8.bin --quant int8

  # 4-bit quantization (packed int4)
  python export_mistral.py --model_dir ../Mistral-7B-v0.1 --out ./mistral-int4.bin --quant int4

---------

This script converts a Hugging Face Mistral model into one standardized binary file that can be fed into the inference engine.

Inputs (from the downloaded model directory):
  - config.json              : model hyperparameters
  - tokenizer.json           : vocabulary
  - model.safetensors.index.json + shard files : weights

Output:
  model.bin with layout:
    [8-byte uint64: size of JSON header]
    [JSON header]: 
        - config
        - vocab/merges
        - tensor info:
            - data type
            - shape
            - tensor start index
            - scales size
            - scales start index
            
    [payload: All tensors as continuous data with quantization scales]
"""

# Per-group symmetric quantization
# Splits tensor in groups, finds the max abs value and computes the scale that maps values -> int range
# With int8 (n_bits=8) the usable range is [-127, 127].
# Returns the quantized tensor and the scales
def quantize(x: torch.Tensor, n_bits: int, group_size: int):
    assert (x.numel() % group_size == 0)

    # Split tensor in groups
    x = x.reshape(-1, group_size)

    # Max int range
    int_max = 2 ** (n_bits - 1) - 1

    # Compute scale for each group
    scales = int_max / x.abs().max(dim=-1).values.unsqueeze(-1)

    # Quantize
    quant = (x * scales).round()

    return quant, scales


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """
    Pack signed 4-bit values into bytes (two per byte).
    q is assumed to be int8 tensor with values in [-8, 7] or [-7, 7].
    Stored as 4-bit two's complement in each nibble.
    """
    # Flatten and ensure int8
    q = q.to(torch.int8).view(-1)

    # If odd number of elements, pad one zero
    if q.numel() % 2 != 0:
        pad = torch.zeros(1, dtype=torch.int8, device=q.device)
        q = torch.cat([q, pad], dim=0)

    # Convert to unsigned nibble representation (two's complement lower 4 bits)
    q_u = (q & 0x0F).to(torch.uint8)

    hi = q_u[0::2]  # even indices
    lo = q_u[1::2]  # odd indices

    packed = (hi << 4) | lo
    return packed  # uint8 tensor, size = ceil(numel/2)


def load_config(header):
    config_path = os.path.join(IN_PATH, "config.json")

    with open(config_path, 'r') as f:
        cfg = json.load(f)
        header["metadata"] = {
            "hidden_size": str(cfg["hidden_size"]),
            "intermediate_size": str(cfg["intermediate_size"]),
            "n_layers": str(cfg["num_hidden_layers"]),
            "n_heads": str(cfg["num_attention_heads"]),
            "n_kv_heads": str(cfg["num_key_value_heads"]),
            "vocab_size": str(cfg["vocab_size"]),
            "max_position_embeddings": str(cfg["max_position_embeddings"]),
            "sliding_window": str(cfg["sliding_window"]),
            "rope_theta": str(cfg["rope_theta"]),
            "norm_eps": str(cfg["rms_norm_eps"]),
            "act_type": cfg["hidden_act"],
            "quant": str(args.quant)
        }

def load_tokenizer(header):
    # Insert the vocab in header["vocab"]
    tokenizer_path = os.path.join(IN_PATH, "tokenizer.json")
    header["tokenizer"] = {}

    with open(tokenizer_path, 'r') as f:
        t = json.load(f)
        header["tokenizer"]["vocab"] = t["model"]["vocab"]
        header["tokenizer"]["merges"] = t["model"]["merges"]

def pad_to_64(offset):
    r = offset % 64
    if r == 0:
        return 0

    return 64 - r


def quant_bytes_for_tensor(numel: int, quant_mode: str) -> int:
    if quant_mode == "int8":
        return numel  # 1 byte per value
    if quant_mode == "int4":
        # 2 values per byte
        return (numel + 1) // 2
    raise ValueError(f"Unsupported quant_mode for bytes: {quant_mode}")

def load_tensor_map(header):
    # Loop through each tensor and add info to header["tensors"]
    header["tensors"] = {}
    start = 0

    for tensor_name in weight_map:
        tensor_file_path = os.path.join(IN_PATH, weight_map[tensor_name])

        with safetensors.safe_open(tensor_file_path, framework="pt") as f:
            tensor = f.get_tensor(tensor_name)
            numel = tensor.numel()

            # Quantize only certain tensors (as before)
            if "_proj" in tensor_name and args.quant != "f32":
                # Quantized tensor
                header["tensors"][tensor_name] = {
                    "dtype": args.quant,                         # "int8" or "int4"
                    "shape": list(tensor.shape)[:4],
                    "offset": start
                }

                # Bytes for quantized weights
                tensor_bytes = quant_bytes_for_tensor(numel, args.quant)
                start += tensor_bytes
                start += pad_to_64(start)

                # Scales: one scale per group of GROUP_SIZE original values
                scale_size = numel // GROUP_SIZE
                header["tensors"][tensor_name]["scale_offset"] = start
                header["tensors"][tensor_name]["scale_size"] = scale_size

                # Scales stored as float32
                start += scale_size * 4
                start += pad_to_64(start)

            # Full float
            else:
                header["tensors"][tensor_name] = {
                    "dtype": "f32",
                    "shape": list(tensor.shape)[:4],
                    "offset": start
                }
                start += numel * 4
                start += pad_to_64(start)

def write_tensor(out, tensor, base_offset, tensor_offset):
        tensor_bytes = tensor.numpy().tobytes()
        out.seek(base_offset + tensor_offset, 0)
        out.write(tensor_bytes)


def write_binary(header):
    with open(OUT_PATH, "wb") as out:
        header_bytes = json.dumps(header).encode("utf-8")
        padding_size = pad_to_64(8 + len(header_bytes))

        header_size = struct.pack("<Q", len(header_bytes) + padding_size)

        out.write(header_size)
        out.write(header_bytes)

        out.seek(padding_size, 1)
        base_offset = out.tell()

        total = len(header["tensors"])
        i = 0
        bar_width = 40

        # Dump all the tensors in the same order as header
        for tensor_name in header["tensors"]:
            i += 1
            print("[" + "#" * int(bar_width * i/total) + "-" * (bar_width - int(bar_width * i/total)) + f"] {int(i/total*100)}%", end="\r")

            tensor_file_path = os.path.join(IN_PATH, weight_map[tensor_name])
            with safetensors.safe_open(tensor_file_path, framework="pt") as f:
                tensor = f.get_tensor(tensor_name)
                scales = None

                if "_proj" in tensor_name and header["tensors"][tensor_name]["dtype"] != "f32":
                    dtype = header["tensors"][tensor_name]["dtype"]  # "int8" or "int4"

                    # Choose bit-width for quantization
                    if dtype == "int8":
                        n_bits = 8
                    elif dtype == "int4":
                        n_bits = 4
                    else:
                        raise ValueError(f"Unknown quant dtype: {dtype}")

                    # Quantize
                    tensor, scales = quantize(tensor, n_bits, GROUP_SIZE)

                    # Quantized values always in int8 container
                    tensor = tensor.to(torch.int8)
                    scales = scales.to(torch.float32)

                    # If int4, pack two 4-bit values per byte
                    if dtype == "int4":
                        tensor = pack_int4(tensor)  # now uint8, length = ceil(numel/2)

                else:
                    tensor = tensor.to(torch.float32)

                write_tensor(out, tensor, base_offset, header["tensors"][tensor_name]["offset"])

                if scales is not None:
                    write_tensor(out, scales, base_offset, header["tensors"][tensor_name]["scale_offset"])



parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--out", default="./model.bin")
parser.add_argument("--quant", default="f32", choices=["f32", "int8", "int4"])
args = parser.parse_args()

IN_PATH = args.model_dir
OUT_PATH = args.out
GROUP_SIZE = 64

# Quantization config
if args.quant == "f32":
    QUANT_BITS = None      # no quantization
elif args.quant == "int8":
    QUANT_BITS = 8
elif args.quant == "int4":
    QUANT_BITS = 4
else:
    raise ValueError(f"Unsupported quant mode: {args.quant}")

# Load weight map
tensor_index_path = os.path.join(IN_PATH, "model.safetensors.index.json")
with open(tensor_index_path, 'r') as f:
    index = json.load(f)
    weight_map = index["weight_map"]

print("\033[1m\033[4mModel Export\033[0m\n"
      f"\033[1mModel Directory:\033[0m {IN_PATH}\n"
      f"\033[1mOutput File:\033[0m     {OUT_PATH}\n"
      f"\033[1mQuantization:\033[0m    {args.quant}\n")

header = {}
load_config(header)
load_tensor_map(header)
load_tokenizer(header)
write_binary(header)

#print(header["tensors"])

print("\nCompleted")

