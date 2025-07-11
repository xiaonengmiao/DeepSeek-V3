import os
from os.path import exists
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


def convert_mtp(hf_ckpt_path, save_path):
    mtp_state_dict = load_file(os.path.join(hf_ckpt_path, "model.safetensors"))
    new_state_dict = {}
    for name in mtp_state_dict.keys():
        param = mtp_state_dict[name]
        if name == "model.embed_tokens.weight":
            new_state_dict["model.layers.61.embed_tokens.weight"] = param
        if name == "model.enorm.weight":
            new_state_dict["model.layers.61.enorm.weight"] = param
        if name == "model.hnorm.weight":
            new_state_dict["model.layers.61.hnorm.weight"] = param
        if name == "model.eh_proj.weight":
            new_state_dict["model.layers.61.eh_proj.weight"] = param
        if name == "model.shared_head_norm.weight":
            new_state_dict["model.layers.61.shared_head.norm.weight"] = param
        if name == "shared_head_head.weight":
            new_state_dict["model.layers.61.shared_head.head.weight"] = param
        if name.startswith("model.layers.0."):
            new_name = "model.layers.61" + name[len("model.layers.0"):]
            new_state_dict[name] = param
    os.mkdirs(save_path, exist_ok=True)
    save_file(new_state_dict, os.path.join(save_path, "model.safetensors"))
    for file_path in []:
        save_dict = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                param: torch.Tensor = f.get_tensor(name)
                if name in ["model.layers.61.self_attn.kv_b_proj.weight"]:
                    save_dict[name] = new_state_dict[name]
                elif name not in ["model.layers.61.self_attn.kv_b_proj.weight_scale", "model.layers.61.self_attn.kv_b_proj.weight_offset"]:
                    save_path[name] = param
        save_file(save_dict, os.path.join(save_path, file_path[-len("quant_model_weight_w8a8_dynamic-00000-of-00162.safetensors")):])

def main(hf_ckpt_path, save_path):
    torch.set_num_threads(8)
    state_dicts = {}

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    param: torch.Tensor = f.get_tensor(name)
                    if name.startswith("model."):
                        name = "model.layers.0" + name[len("model.layers.61"):]
                    if "embed_tokens" in name or "enorm" in name or "hnorm" in name or "eh_proj" in name:
                        name = name.replace("layers.0.", "")
                    if "shared_head.norm" in name:
                        name = name.replace("layers.0.shared_head.", "")
                    if "shared_head.head" in name:
                        name = name.replace("model.layers.0.shared_head.head", "lm_head")
                    state_dicts[name] = param

    os.makedirs(save_path, exist_ok=True)

    save_file(state_dicts, os.path.join(save_path, f"model-mtp.safetensors"))

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    main(args.hf_ckpt_path, args.save_path)
