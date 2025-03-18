import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from model_wrapper import ModelForSC, ModelForSCDual, ModelForSCProbing, ModelForSCDualProbing
from dataset import LRADataset
import torch
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import math
import itertools
import torch.nn.init as init

from megatron.arguments import parse_args
from megatron.initialize import initialize_megatron

from calflops import calculate_flops

import sys

from torch import profiler

import wandb
wandb.init(
    project="transformer-LRA-offical",
    config={
        "architecture": "transformer",
        "dataset": "LRA",
    }
)

use_XMoE = True

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model", dest="model", required=True)
parser.add_argument("--task", type=str, help="task", dest="task", required = False)
parser.add_argument("--skip_train", type = int, help = "skip_train", dest = "skip_train", default = 0)
parser.add_argument("--logging", action='store_true', default=False)
parser.add_argument("--expname", type=str, default="default")

# Model configs
parser.add_argument("--attention_grad_checkpointing", default=False, action="store_true")
parser.add_argument("--num_landmarks", default=128, type=int)
parser.add_argument("--window_size", default=129, type=int)
parser.add_argument("--conv_kernel_size", default=-1, type=int)
parser.add_argument("--learn_pos_emb", default=1, type=int,
                    help="Use 0 or 1 to represent false and true")
parser.add_argument("--tied_weights", default=False, action="store_true")
parser.add_argument("--embedding_dim", default=64, type=int)
parser.add_argument("--transformer_dim", default=64, type=int)
parser.add_argument("--transformer_hidden_dim", default=128, type=int)
parser.add_argument("--head_dim", default=32, type=int)
parser.add_argument("--num_heads", default=2, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--vocab_size", default=512, type=int)
parser.add_argument("--max_seq_len", default=4096, type=int)
parser.add_argument("--dropout_prob", default=0.1, type=float)
parser.add_argument("--attention_dropout", default=0.1, type=float)
parser.add_argument("--pooling_mode", default="MEAN", type=str)
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--cls_token", default=False, action='store_true')
parser.add_argument("--random_rate", default=0.2, type=float)

#additional configs
parser.add_argument("--attn_n_experts", default=2, type=int)
parser.add_argument("--attn_topk", default=1, type=int)
parser.add_argument("--n_experts", default=4, type=int)
parser.add_argument("--topk", default=2, type=int)
parser.add_argument("--auto_group", action='store_true', default=False)
parser.add_argument("--n_patches", default=16, type=int)
parser.add_argument("--capacity", default=1, type=float)
parser.add_argument("--epsilon", default=1.0, type=float)
# parser.add_argument("--expert_dim", default=32, type=int)
parser.add_argument("--init_method", default=init.xavier_normal_)
parser.add_argument("--output_layer_init_method", default=init.xavier_normal_)

# Training configs
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--warmup", default=8000, type=int)
parser.add_argument("--lr_decay", default="linear", type=str)
parser.add_argument("--fixed_lr", default=False, action='store_true')
parser.add_argument("--weight_decay", default=0.001, type=float)
parser.add_argument("--adam_eps", default=1e-6, type=float)
parser.add_argument("--eval_frequency", default=500, type=int)
parser.add_argument("--num_train_steps", default=60000, type=int)
parser.add_argument("--num_eval_steps", default=1562, type=int)
parser.add_argument("--fp32_attn", default=True, action='store_true')
parser.add_argument("--conv_zero_init", default=False, action='store_true')

# Dataset Configs
parser.add_argument("--n_train_samples", default=25000, type=int)
parser.add_argument("--n_dev_samples", default=25000, type=int)
parser.add_argument("--n_test_samples", default=25000, type=int)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--cls_last_layer", default=False, action='store_true')
parser.add_argument("--seed", default=1234, type=int)
parser.add_argument("--linformer_k", default=256, type=int)
parser.add_argument("--rp_dim", default=256, type=int)
parser.add_argument("--num_hash", default=2, type=int)
parser.add_argument("--chk_path", default="LRA_chks", type=str)
parser.add_argument("--test_flops", default=False, action='store_true')

if use_XMoE:
    parser = parse_args(parser)
    args = parser.parse_args()
    if use_XMoE:
        args.rank = int(os.getenv('RANK', '0'))
        args.world_size = int(os.getenv("WORLD_SIZE", '1'))
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        args.fp16 = False
        args.hidden_size = args.transformer_dim
        initialize_megatron(args)
else:
    args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

# args.test_flops = True
args.attn_type = args.model
args.mixed_precision = False
task = args.task

checkpoint_dir = args.chk_path
print(args)
device_ids = [0]

if task == "retrieval":
    if args.test_flops:
        model = ModelForSCDualProbing(args)
    else:
        model = ModelForSCDual(args)
else:
    if args.test_flops:
        model = ModelForSCProbing(args)
    else:
        model = ModelForSC(args)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush=True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush=True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

data_path = 'datasets'

ds_iter = {
    "train":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.train.pickle", True, True if task == "image" else False), batch_size=args.batch_size, drop_last=True)),
    "dev":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.dev.pickle", True, False), batch_size=args.batch_size, drop_last=True)),
    "test":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.test.pickle", True, False), batch_size=args.batch_size, drop_last=False)),
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.999), eps=args.adam_eps, weight_decay=args.weight_decay, amsgrad=True
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=args.learning_rate,
    pct_start=args.warmup / args.num_train_steps,
    anneal_strategy=args.lr_decay,
    total_steps=args.num_train_steps
)

amp_scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

def step(component, step_idx, is_train):
    t0 = time.time()

    optimizer.zero_grad()

    _, batch = next(ds_iter[component])
    for key in batch:
        batch[key] = batch[key].cuda()

    if (args.model == 'nystrom' or args.model == 'reformer') and args.pooling_mode.lower() == 'cls':
        for key in batch:
            if 'input_ids' in key or 'mask' in key:
                batch[key] = batch[key][:, :-1].contiguous()

    if component == "train":
        outputs = {}

        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                partial_inputs_list[idx][key] = inp
        torch.cuda.empty_cache()
        for partial_inputs in partial_inputs_list:
            if args.test_flops:
                if 'input_ids_1' in partial_inputs:
                    flops = FlopCountAnalysis(
                        model, [partial_inputs['input_ids_0'][:1], partial_inputs['input_ids_1'][:1],
                                partial_inputs['mask_0'][:1], partial_inputs['mask_1'][:1], partial_inputs['label'][:1]])
                    pass
                else:
                    #Note: To enable calflops to support list input, it is necessary to comment on lines 153 and 154
                    # of the source code flops_comounter file:
                    # for key, value in kwargs.items():
                    #     kwargs[key] = value.to(device)
                    kwargs = {'input_list': [partial_inputs['input_ids_0'][:1], partial_inputs['mask_0'][:1], partial_inputs['label'][:1]]}
                    flops, macs, params = calculate_flops(model=model,
                                                          kwargs=kwargs,
                                                          output_as_string=True,
                                                          output_precision=4)
                exit()

            partial_outputs = model(**partial_inputs)
            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]
            partial_outputs["loss"].backward()

        optimizer.step()
        optimizer.zero_grad()

        if (not args.fixed_lr) or step_idx < args.warmup:
            lr_scheduler.step()
    else:
        with torch.no_grad():
            outputs = {}

            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]

    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t

    if(is_train == False):
        wandb.log({"valid_loss": loss, "valid_accuracy": accu})

    print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)

def print_summary(summary, save_if_improved, train_step_idx, subset):
    # subset: str, the subset to report the result
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])

    print()
    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save({"model_state_dict":model.module.state_dict()}, log_f_path.replace(".log", ".model"))
            print(f"best_accu={best_accu}. Saved best model")

    summary_round = {"train_step_idx":train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key+f"_{subset}"] = summary[key]
        else:
            summary_round[key+f"_{subset}"] = round(summary[key], 4)

    print(summary_round, flush=True)
    log_f.write(json.dumps(summary_round, sort_keys = True) + "\n")
    log_f.flush()

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []

init_t = time.time()

log_f_path = os.path.join(checkpoint_dir, f"{args.expname}_output.log")
log_f = open(log_f_path, "a+")

summary = {
    component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
    for component in ["train", "dev", "test"]
}

accumu_steps = max(args.batch_size // len(device_ids) // 32, 1)
if args.skip_train == 0:
    try:
        model.train()
        for train_step_idx in range(args.num_train_steps):
            outputs = step("train", train_step_idx, is_train=True)

            if (train_step_idx + 1) % args.eval_frequency == 0:
                print_summary(summary["train"], False, train_step_idx, 'train')
                model.eval()
                with torch.no_grad():
                    for dev_step_idx in range(args.num_eval_steps):
                        outputs = step("test", dev_step_idx, is_train=False)
                print_summary(summary["test"], True, train_step_idx, 'test')
                model.train()
    except KeyboardInterrupt as e:
        print(e)

checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location="cpu")
model.module.load_state_dict(checkpoint["model_state_dict"])