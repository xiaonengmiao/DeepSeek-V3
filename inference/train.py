import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import math
from typing import Any, Dict, List

import numpy as np
import peft
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from model import MTP, ModelArgs
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          get_linear_schedule_with_warmup)

torch.backends.cuda.matmul.allow_tf32 = True

train_config = {
    "lr": 3e-5,
    "bs": 1,
    "gradient_accumulation_steps": 4,
    "datapath": "data/train.json",
    "is_warmup": True,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": 2,
    "embedding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2500,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": "/data/l84248763/deepseek-mtp",
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 100,
    "cpdir": "/home/savedir"
}

set_seed(42)
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=train_config["gradient_accumulation_steps"])
if accelerator.is_main_process:
    import wandb
    wandb.init(project="MTP", entity="haohan", config=train_config)


class CustomDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        super().__init__()
        self.data = data_path
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        data = torch.load(self.data[idx])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data['loss_mask'][:train_config["max_len"]][None, :]

        length = hidden_state.shape[1]
        # length_q = data['length_q'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.zeros([[0]])
        input_ids_target = torch.cat([input_ids_target, zeropadding], dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat([target, zeropadding], dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target

        if self.transform:
            new_data = self.transform(new_data)
        
        return new_data
    

class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat([intensors, padding_tensor], dim=1)
        return outtensors
    
    def paddingtensors2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat([intensors, padding_tensor], dim=1)
        return outtensors
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensors2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features]
        )
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features]
        )
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "loss_mask": batch_loss_mask,
            "attention_mask": batch_attention_mask
        }
        return batch
    

class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        noise = torch.normal(mean=self.mean, std=self.std, size=sample['hidden_state_big'].shape)
        sample['hidden_state_big'] += noise
        return sample


class AddUniformNoise:
    def __init__(self, std: float = 0.0):
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        noise = (torch.rand_like(sample['hidden_state_big']) - 0.5) * self.std * 512 / sample['hidden_state_big'].shape[1]
        sample['hidden_state_big'] += noise
        return sample


def top_accuracy(output, target, topk=(1,)):
    # output.shape: (batch_size, num_classes), target.shape: (batch_size,)
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def compute_loss(target, target_p, predict, loss_mask):
    out_head = head(predict)
    out_logp = torch.nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, dim=2)) / (loss_mask.sum() + 1e-5)
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, dim=2)) / (loss_mask.sum() + 1e-5)
    return vloss, ploss, out_head

def list_files(path):
    datapath = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    elif train_config["noise"] == "gaussian":
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
    else:
        raise ValueError("Unsupported noise type. Use 'uniform' or 'gaussian'.")
else:
    aug = None


datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

traindataset = CustomDataset(traindatapath, transform=aug)
testdatapath = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True, collate_fn=DataCollatorWithPadding(),
                          num_workers=train_config["num_workers"], pin_memory=True)
test_loader = DataLoader(testdatapath, batch_size=train_config["bs"], shuffle=False, collate_fn=DataCollatorWithPadding(),
                          num_workers=train_config["num_workers"], pin_memory=True)
if accelerator.is_main_process:
    if not os.path.exists(train_config["cpdir"]):
        os.makedirs(train_config["cpdir"])
baseconfig = AutoConfig.from_pretrained(train_config["config_path"])

with open(os.path.join(train_config["config_path"], "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
device_map = {}
for name in index_json["weight_map"].keys():
    if ".experts." in name:
        sp = int(name.split(".")[5])
        if sp < 86:
            device_map[name] = 1
        elif sp < 172:
            device_map[name] = 2
        else:
            device_map[name] = 3
    elif "self_attn" in name:
        device_map[name] = 4
    else:
        device_map[name] = 5
with open(os.path.join(train_config["config_path"], "device_map.json"), "w") as f:
    json.dump(device_map, f)

with open(os.path.join(train_config["config_path"], "device_map.json"), "r") as f:
    device_map = json.load(f)

head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

with open(os.path.join(train_config["config_path"], "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["shared_head_head.weight"]
with safe_open(os.path.join(train_config["config_path"], head_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("shared_head_head.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim].float()

head.weight.data = tensor
head = head.cuda()
head.eval()

for param in head.parameters():
    param.requires_grad = False

model = AutoModelForCausalLM.from_pretrained(
    train_config["config_path"],  
    device_map=device_map, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
target_modules = ["eh_proj"]
config = peft.LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=peft.TaskType.CAUSAL_LM,
    target_modules=target_modules
)
model = peft.get_peft_model(model, config)
model.print_trainable_parameters()

criterion = torch.nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    schedular = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    model, optimizer, train_loader, test_loader, schedular = accelerator.prepare(
        model, optimizer, train_loader, test_loader, schedular
    )
else:
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

for epoch in range(num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            input_ids = data["input_ids"].cuda()
            hidden_states = data["hidden_states"].cuda()
            target = data["target"].cuda()
            loss_mask = data["loss_mask"].cuda()
            attention_mask = data["attention_mask"].cuda()

            predict = model(input_ids, attention_mask=attention_mask, previous_hidden_states=hidden_states,
                            output_hidden_states=True, return_dict=True).hidden_states[-1]
            with torch.no_grad():
                target_head = head(target)
                target_p = torch.nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
            loss_mask = loss_mask[:, :, None]
            vloss, ploss, out_head = compute_loss(target, target_p, predict, loss_mask)

            loss = train_config["p_w"] * ploss + train_config["v_w"] * vloss
            accelerator.backward(loss)
            nn.utils.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                schedular.step()
        
        with torch.no_grad():
            _, predicted = torch.max(out_head, dim=2)
            _, targeted = torch.max(target_head, dim=2)
            ct = loss_mask.sum().item()
            cc = ((predicted == targeted) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            targeted = targeted.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, targeted, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f"train/top_{i + 1}_acc"] = topkacc[id].item() / ct
            wandb.log(logdict)
        
        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1
    
    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss = epoch_loss / num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / (total + 1e-5)})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
        wandb.log({
            'train/epochloss': epoch_loss,
            'train/epochacc': correct / (total + 1e-5)
        })
    if (epoch + 1) % train_config["save_freq"] == 0 and accelerator.is_local_main_process:
        print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Test Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
        wandb.log({
            'test/epochloss': epoch_loss,
            'test/epochacc': correct / (total + 1e-5)
        })
        accelerator.save_state(os.path.join(train_config["cpdir"], f"state_{epoch}"))