# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    get_scheduler,
    GenerationConfig
)

from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from torch.utils.data import DataLoader

from torch.optim import AdamW
from accelerate import Accelerator

import wandb
import gc

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_len: int = field(
        default=512,
        metadata={"help": "Maximum length of prompt, rejected and chosen response. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='oasst1_dpo',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    beta: float = field(default=0.01, metadata={"help": 'KL penalty'})
    num_epochs: int = field(default=1, metadata={"help": 'How many epochs to do'})

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_models(args, checkpoint_dir):

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="left",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama', ## I am always using the llama tokenizer ## if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        use_auth_token=args.use_auth_token,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if checkpoint_dir is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
    else:
        print(f'adding LoRA modules...')
        modules = find_all_linear_names(args, model)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                print(module.weight.dtype)

    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'embed_tokens' in name:
                print('Embedding is trained')
            elif 'lm_head' in name:
                print('Unembedding is trained')
        if 'embed_tokens' in name:
            print(param.requires_grad)
            print(param.grad)
        if 'lm_head' in name:
            print(param.requires_grad)
            print(param.grad)
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel
    ):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
        
def load_and_split_data(args):
     # Load dataset.
    if args.dataset == 'oasst1_dpo_1':
        dataset = load_dataset("json", data_files={"train": 'data/dpo/oasst1_dpo_train_1.json', "eval" : 'data/dpo/oasst1_dpo_test_1.json'})
    train_dataset = dataset['train']
    eval_dataset = dataset['eval']

    return train_dataset, eval_dataset

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        # if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def dpo_loss(pi_yw_logps, pi_yl_logps, ref_yw_logps, ref_yl_logps, beta):

    pi_logratios = pi_yw_logps - pi_yl_logps

    ref_logratios = ref_yw_logps - ref_yl_logps

    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards_chosen = beta * (pi_yw_logps - ref_yw_logps).detach().to('cpu')
    rewards_rejected = beta * (pi_yl_logps - ref_yl_logps).detach().to('cpu')

    return losses, rewards_chosen, rewards_rejected

def pad_to_length(input, max_length, padding_value):
    
    p = padding_value*torch.ones(size=(input.shape[0], max_length - input.shape[-1]), dtype=torch.int64)

    return torch.cat((input, p), -1)


def prepare_batch(batch, tokenizer, max_length):

    b = {k: tokenizer(v, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt", max_length=max_length) for k, v in batch.items() if k not in ['lang', 'parent_id']}

    chosen_response = {s : torch.cat((b['prompt'][s], b['chosen'][s]), -1) for s in b['chosen']}
    rejected_response = {s : torch.cat((b['prompt'][s], b['rejected'][s]), -1) for s in b['rejected']}
    b['chosen'] = chosen_response
    b['rejected'] = rejected_response
    l_p = b['prompt']['input_ids'].shape[-1]
    chosen_labels = chosen_response['input_ids'].clone()
    chosen_labels[0, :l_p] = -100
    rejected_labels = rejected_response['input_ids'].clone()
    rejected_labels[0, :l_p] = -100
    b['chosen_labels'] = chosen_labels
    b['rejected_labels'] = rejected_labels

    l_max = max(b['chosen']['input_ids'].shape[-1], b['rejected']['input_ids'].shape[-1])

    concatenated = {}
    concatenated['response'] = {s : torch.cat((
                            pad_to_length(b['chosen'][s], l_max, padding_value=(tokenizer.pad_token_id if s == 'input_ids' else 0)), 
                            pad_to_length(b['rejected'][s], l_max, padding_value=(tokenizer.pad_token_id if s == 'input_ids' else 0))), 0) 
                            for s in b['chosen']}
    concatenated['labels'] = torch.cat((
                            pad_to_length(b['chosen_labels'], l_max, padding_value=tokenizer.pad_token_id), 
                            pad_to_length(b['rejected_labels'], l_max, padding_value=tokenizer.pad_token_id)), 0)

    return concatenated
                
def compute_step(batch, model, tokenizer, accelerator, beta, max_length):

    bs = len(batch['prompt'])

    concatenated = prepare_batch(batch, tokenizer, max_length)

    fwd = model(**concatenated['response'], use_cache=False)['logits'][:, :-1, :]
    labels = concatenated['labels'][:, 1:].clone()
    mask = labels != -100
    labels[labels == -100] = 0

    pi_logps = torch.gather(F.log_softmax(fwd, dim=-1), 2, labels.unsqueeze(2)).squeeze(2)

    pi_yw_logps = (pi_logps * mask).sum(-1)[:bs]
    pi_yl_logps = (pi_logps * mask).sum(-1)[bs:]

    with torch.no_grad():
        with accelerator.unwrap_model(model).disable_adapter():
            fwd = model(**concatenated['response'], use_cache=False)['logits'][:, :-1, :]

            ref_logps = torch.gather(F.log_softmax(fwd, dim=-1), 2, labels.unsqueeze(2)).squeeze(2)
            
            ref_yw_logps = (ref_logps * mask).sum(-1)[:bs]
            ref_yl_logps = (ref_logps * mask).sum(-1)[bs:]
    
    losses, rewards_chosen, rewards_rejected = dpo_loss(pi_yw_logps, pi_yl_logps, ref_yw_logps, ref_yl_logps, beta)

    loss = losses.mean()
    r_c = rewards_chosen.mean()
    r_r = rewards_rejected.mean()
    r_a = (rewards_chosen > rewards_rejected).float().mean()
    r_d = (rewards_chosen - rewards_rejected).mean()

    return loss, [r_c, r_r, r_a, r_d]

def epoch(s, model, accelerator, loader_t, loader_e, tokenizer, optimizer, scheduler, beta, max_length):
    for step, batch in enumerate(loader_t):
        step += s

        model.train()
        with accelerator.accumulate(model):
            loss, r = compute_step(batch=batch,
                                    model=model,
                                    tokenizer=tokenizer,
                                    accelerator=accelerator,
                                    beta=beta,
                                    max_length=max_length
                                    )

            accelerator.print('step', step, 'loss', loss, 'average rewards chosen', r[0].item(), 'average rewards rejected', r[1].item())
            
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            accelerator.log({
                    'DPO loss': loss.item(), 
                    'learning rate': scheduler.get_last_lr()[0],
                    'average train rewards chosen': r[0].item(), 
                    'average train rewards rejected': r[1].item(),
                    'average train rewards acc': r[2].item(),
                    'average train rewards margin': r[3].item(),
                    'epochs': step / len(loader_t)
                    },
                    step=step
                    )

            del loss, r
            gc.collect()
        
        if step % 20 == 0 and step > 0:
            accelerator.print("Evaluating")
            r_e_av = [0]*4
            loss_e_av = 0
            for batch_e in loader_e:
                model.eval()
                with torch.no_grad():
                    loss_e, r_e = compute_step(batch=batch_e,
                                                      model=model,
                                                      tokenizer=tokenizer,
                                                      accelerator=accelerator,
                                                      beta=beta)
                    loss_e_av += loss_e.mean() / len(loader_e)

                    r_e_av = [r_e_av[i] + r_e[i].mean() / len(loader_e) for i in range(4)]
            
            accelerator.log({
                    'DPO test loss': loss_e_av, 
                    'average test rewards chosen': r_e_av[0], 
                    'average test rewards rejected': r_e_av[1],
                    'average test rewards acc': r_e_av[2],
                    'average test rewards margin': r_e_av[3],
                    },
                    step=step
                    )

    return step

def train():

    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model, tokenizer = get_models(args, checkpoint_dir)

    model.config.use_cache = False
    print('loaded models')
    set_seed(args.seed)

    batch_size_t = args.per_device_train_batch_size
    batch_size_e = args.per_device_eval_batch_size
    train_ds, eval_ds = load_and_split_data(args=args)
    
    max_number_examples = args.max_train_samples

    if max_number_examples is not None:
        train_ds = train_ds.select(range(max_number_examples))

    train_loader = DataLoader(train_ds, batch_size=batch_size_t, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size_e, pin_memory=True)

    beta = args.beta

    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = args.warmup_ratio * num_training_steps

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = get_scheduler(args.lr_scheduler_type, 
                              optimizer=optimizer, 
                              num_warmup_steps=num_warmup_steps, 
                              num_training_steps=num_training_steps,
                            )
    
    accelerator = Accelerator(mixed_precision="bf16", 
                              log_with=args.report_to, 
                              gradient_accumulation_steps=args.gradient_accumulation_steps
                            )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    print('training started')

    accelerator.init_trackers(project_name="QLoRA + DPO",
                        config={"beta": beta, 
                                "learning rate": args.learning_rate, 
                                "batch size train": batch_size_t,
                                "batch size eval": batch_size_e,
                                "epochs": num_epochs 
                                }
                     )
    s = 0
    for _ in range(num_epochs):

        s = epoch(s=s,
                model=model, 
                accelerator=accelerator,
                loader_t=train_loader,
                loader_e=eval_loader,
                tokenizer=tokenizer, 
                optimizer=optimizer,
                scheduler=scheduler, 
                beta=beta,
                max_length=args.max_len
                )

    accelerator.end_training()

    print('Saving PEFT adapter model...')

    peft_model_path = os.path.join(args.output_dir, "adapter_model")
    model.save_pretrained(peft_model_path)


if __name__ == "__main__":
    train()