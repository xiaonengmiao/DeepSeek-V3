import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


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
