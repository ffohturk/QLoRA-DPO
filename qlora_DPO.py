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
import pandas as pd

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

import wandb

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
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
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
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

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


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_models(args, checkpoint_dir):

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map={"" : 0},
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

    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": 1},
        torch_dtype=torch.bfloat16,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        use_auth_token=args.use_auth_token,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
            model_ref=model_ref
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
        # if 'lm_head' in name or 'embed_tokens' in name:
        #     if hasattr(module, 'weight'):
        #         if args.bf16 and module.weight.dtype == torch.float32:
        #             module = module.to(torch.bfloat16)
        #         print(module.weight.dtype)

    return model, model_ref, tokenizer

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
        # if 'q_proj' in name:
        #     print('-----')
        #     print(name)
        #     print(param.requires_grad)
        #     print(param.grad)
        #     print('-----')
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    model_ref: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    model_ref.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

        input_embeddings_data_ref = model_ref.get_input_embeddings().weight.data
        output_embeddings_data_ref = model_ref.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data_ref[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data_ref[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data_ref[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data_ref[-num_new_tokens:] = output_embeddings_avg
        
def load_and_split_data(args):
     # Load dataset.
    dataset = load_dataset("json", data_files={"train": 'data/dpo/anthropic_DPO_train.json', "eval" : 'data/dpo/anthropic_DPO_test.json'})

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        # if args.group_by_length:
        #     eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

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

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):

    """
    pi_logps: policy logprobs, shape (B,)
    ref_logps: reference model logprobs, shape (B,)
    yw_idxs: preferred completion indices in [0, B-1], shape (T,)
    yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
    beta: temperature controlling strength of KL penalty
    Each pair of (yw_idxs[i], yl_idxs[i]) represents the
    indices of a single preference pair.
    """

    pi_yw_logps = torch.gather(pi_logps, -1, yw_idxs.unsqueeze(-1).to('cuda:0')).squeeze(-1)
    pi_yl_logps = torch.gather(pi_logps, -1, yl_idxs.unsqueeze(-1).to('cuda:0')).squeeze(-1)
    ref_yw_logps = torch.gather(ref_logps, -1, yw_idxs.unsqueeze(-1).to('cuda:1')).squeeze(-1)
    ref_yl_logps = torch.gather(ref_logps, -1, yl_idxs.unsqueeze(-1).to('cuda:1')).squeeze(-1)

    pi_logratios = pi_yw_logps.sum(dim=-1) - pi_yl_logps.sum(dim=-1)
    ref_logratios = ref_yw_logps.sum(dim=-1) - ref_yl_logps.sum(dim=-1)

    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios.to('cuda:0')))
    rewards = beta * (pi_logps - ref_logps.to('cuda:0')).detach().to('cpu')
    
    return losses, rewards

def generate(input, model, max_new_tokens):
    device = input['input_ids'].device
    log_pp = torch.tensor([]).to(device)
    for i in range(max_new_tokens):
        fwd = model(**input, use_cache=False)['logits'][0, -1]
        log_p = F.log_softmax(fwd, dim=-1)
        next_tok = torch.argmax(log_p).unsqueeze(0)
        input['input_ids'] = torch.cat((input['input_ids'].squeeze(0), next_tok), 0).unsqueeze(0)
        input['attention_mask'] = torch.ones_like(input['input_ids'])
        log_pp = torch.cat((log_pp, log_p.unsqueeze(0)), 0)

    return log_pp


def epoch(model, model_ref, loader, tokenizer, optimizer, scheduler, max_new_tokens, beta, status='train'):

    for batch in loader:
        
        batch_size = len(batch['prompt'])

        b = {k: tokenizer(v, return_tensors="pt", padding=True, truncation=True, max_length=512) for k, v in batch.items()}
        
        # max_new_tokens = min(b['chosen']['input_ids'].shape[-1], b['rejected']['input_ids'].shape[-1])

        pi_logps = torch.tensor([]).to('cuda:0')
        ref_logps = torch.tensor([]).to('cuda:1')
        for i in range(batch_size):
            input = tokenizer(batch['prompt'][i], return_tensors="pt", truncation=True, max_length=512).to('cuda:0')
            log_p = generate(input=input, model=model, max_new_tokens=max_new_tokens)
            pi_logps = torch.cat((pi_logps, log_p.unsqueeze(0)), 0)

            with torch.no_grad():
                log_p = generate(input=input.to('cuda:1'), model=model_ref, max_new_tokens=max_new_tokens)
                ref_logps = torch.cat((ref_logps, log_p.unsqueeze(0)), 0)
            
        # print('pi logs', pi_logps)
        # print('ref logs', ref_logps)

        # TODO
        # Deal with the fact that when we generate in the reference model, you could end up generating the EOS before max_new_tokens is reached. 
        # What do you do in that case? Here we just go on, but you could also truncate all squence lenghts to the shortest one?

        # print('pi_logps grad', pi_logps.requires_grad)
        # print('ref_logps grad', ref_logps.requires_grad)

        yw_idxs = b['chosen'].input_ids[:, :max_new_tokens]
        yl_idxs = b['rejected'].input_ids[:, :max_new_tokens]

        # print(pi_logps.shape)
        # print(yw_idxs.shape)

        losses, rewards = dpo_loss(pi_logps=pi_logps, ref_logps=ref_logps, yw_idxs=yw_idxs, yl_idxs=yl_idxs, beta=beta)

        loss = losses.mean(dim=0)

        print('loss', loss, 'av rewards', rewards.mean(dim=0)) # Reward is shaped as (new tokens, vocab size)

        if status == 'train':
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

        wandb.log({'loss_dpo': loss, 'learning rate': scheduler.get_last_lr()[0]})

        del pi_logps
        del losses
        del rewards 
        del loss
        del ref_logps

def train():

    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model, model_ref, tokenizer = get_models(args, checkpoint_dir)
    print('Parametrized model loaded on', model.device)
    print(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0))
    print('Reference model loaded on', model_ref.device)
    print(torch.cuda.memory_reserved(1) - torch.cuda.memory_allocated(1))

    model_ref.eval()

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    batch_size = 8
    train_ds, eval_ds = load_and_split_data(args=args)

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size)

    beta = 0.01

    num_epochs = 1 
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = args.warmup_ratio * num_training_steps

    max_new_tokens = 50

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = get_scheduler("cosine", 
                              optimizer=optimizer, 
                              num_warmup_steps=num_warmup_steps, 
                              num_training_steps=num_training_steps,
                              )
    print('training started')

    run = wandb.init(project="QLoRA + DPO",
                     config={"beta": beta, 
                             "learning rate": args.learning_rate, 
                             "max new tokens": max_new_tokens,
                             "batch size": batch_size,
                             }
                     )

    for _ in range(num_epochs):

        model.train()
        epoch(model=model, 
              model_ref=model_ref, 
              loader=train_loader, 
              tokenizer=tokenizer, 
              optimizer=optimizer,
              scheduler=scheduler, 
              max_new_tokens=max_new_tokens,
              beta=beta, 
              status='train',
              )
        
        model.eval()
        epoch(model=model, 
              model_ref=model_ref, 
              loader=eval_loader, 
              tokenizer=tokenizer, 
              optimizer=optimizer, 
              scheduler=scheduler,
              max_new_tokens=max_new_tokens,
              beta=beta, 
              status='eval',
              )

    wandb.finish()

if __name__ == "__main__":
    train()