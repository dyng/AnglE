# -*- coding: utf-8 -*-

import os
import argparse
import random

import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from angle_emb import AnglE, AngleDataTokenizer

from emb_model import EmbModel
from mteb import MTEB

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Specify pretrained_model_path, default None')
    parser.add_argument('--pretrained_lora_path', type=str, default=None, help='Specify pretrained_lora_path, default None')
    parser.add_argument('--bellm_class_name', type=str, default=None, help='Specify bellm class name, default None')
    parser.add_argument('--train_name_or_path', type=str, required=True, help='Specify train_name_or_path, required')
    parser.add_argument('--train_subset_name', type=str, default=None, help='Specify train_subset_name, required')
    parser.add_argument('--prompt', type=str, default=None, help='Specify prompt')
    parser.add_argument('--save_dir', type=str, default=None, help='Specify save dir, default None')
    parser.add_argument('--seed', type=int, default=42, help='Specify random seed, default 42')
    parser.add_argument('--on_device', type=str, default=None, help='Specify training device, default None means using best device')
    parser.add_argument('--dataset_seed', type=int, default=None, help='Specify random dataset_seed, default None')
    parser.add_argument('--workers', type=int, default=25, help='Specify dataset workers, default None')
    parser.add_argument('--w1', type=float, default=1.0, help='Specify w1 (cosine), default 1.0')
    parser.add_argument('--w2', type=float, default=35.0, help='Specify w2 (ibn), default 1.0')
    parser.add_argument('--w3', type=float, default=1.0, help='Specify w3 (angle), default 1.0')
    parser.add_argument('--angle_tau', type=float, default=1.0, help='Specify angle_tau, default 1.0')
    parser.add_argument('--cosine_tau', type=float, default=20.0, help='Specify cosine_tau, defaut 20.0')
    parser.add_argument('--ibn_tau', type=float, default=20.0, help='Specify ibn_tau, defaut 20.0')
    parser.add_argument('--is_llm', type=int, default=0, choices=[0, 1], help='Specify is_llm, defaut 1')
    parser.add_argument('--apply_lora', type=int, default=0, choices=[0, 1], help='Specify apply_lora, defaut 0')
    parser.add_argument('--load_kbit', type=int, default=None, choices=[4, 8, 16], help='Specify load_kbit, default None')
    parser.add_argument('--lora_r', type=int, default=32, help='Specify lora_r, defaut 32')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora_alpha, defaut 32')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora_dropout, defaut 0.1')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify learning_rate, defaut 1e-5')
    parser.add_argument('--start_bilayer_index', type=int, default=None, help='Specify start_bilayer_index, defaut None')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Specify warmup_steps, defaut 100')
    parser.add_argument('--logging_steps', type=int, default=100, help='Specify logging_steps, defaut 100')
    parser.add_argument('--pooling_strategy', type=str, default='cls',
                        help='Specify pooling_strategy from [`avg`, `cls`, `cls_avg`, `first_last_avg`]')
    parser.add_argument('--epochs', type=int, default=10, help='Specify epochs, default 10')
    parser.add_argument('--save_steps', type=int, default=1000, help='Specify save_steps, default 1000')
    parser.add_argument('--batch_size', type=int, default=32, help='Specify batch size, default 32')
    parser.add_argument('--maxlen', type=int, default=512, help='Specify max length, default 512')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Specify gradient_accumulation_steps, default 1')
    parser.add_argument('--torch_dtype', type=str, default=None, help='Specify torch_dtype, default 1')
    parser.add_argument('--fp16', type=bool, default=None, choices=[0, 1], help='Specify fp16, default None')
    parser.add_argument('--compute_similar_matrix', type=int, default=1, choices=[0, 1], help='Specify compute_similar_matrix, default 1')
    parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
    parser.add_argument('--wandb_api_key', type=str, default=None,help='Specify wandb_api_key, default None')
    parser.add_argument('--hub_model_id', type=str, default=None, help='Specify push_to_hub_model_id, default None, format like organization/model_id')
    parser.add_argument('--model_name', type=str, default='roberta-large',
                        help='Specify model_name, default roberta-large')
    parser.add_argument('--run_eval', type=bool, default=0, choices=[0, 1], help='Specify if need to test against MTEB, default 0')
    args = parser.parse_args()
    print('Args:', args)
    return args


def train(args):
    if args.seed is not None and args.seed > 0:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.wandb_api_key is None:
        os.environ['WANDB_DISABLED'] = 'true'
    else:
        import wandb
        wandb.login(key=args.wandb_api_key)

    model = AnglE(args.model_name,
                  max_length=args.maxlen,
                  pretrained_model_path=args.pretrained_model_path,
                  pretrained_lora_path=args.pretrained_lora_path,
                  pooling_strategy=args.pooling_strategy,
                  train_mode=True,
                  device=args.on_device,
                  is_llm=args.is_llm,
                  apply_lora=args.apply_lora,
                  lora_config_kwargs={
                      'r': args.lora_r,
                      'lora_alpha': args.lora_alpha,
                      'lora_dropout': args.lora_dropout,
                      'target_modules': ['fc2', 'Wqkv', 'fc1'] if 'BePhi2Model' == args.bellm_class_name else None,
                  },
                  load_kbit=args.load_kbit,
                  bellm_class_name=args.bellm_class_name,
                  kbit_kwargs={'use_gradient_checkpointing': False} if 'BePhi2Model' == args.bellm_class_name else None,
                  torch_dtype=args.torch_dtype)


    if args.start_bilayer_index is not None:
        model.backbone.set_start_bilayer_index(args.start_bilayer_index)

    if os.path.exists(args.train_name_or_path):
        ds = load_dataset('json', data_files=[args.train_name_or_path])
    else:
        ds = load_dataset(args.train_name_or_path, args.train_subset_name)

    if 'validation' not in ds:
        train_test_split = ds['train'].train_test_split(test_size=0.2)  # 80% train, 20% validation
        ds['train'] = train_test_split['train']
        ds['validation'] = train_test_split['test']

    train_ds = prepare_contrastive_dataset(ds['train'], model)
    valid_ds = prepare_contrastive_dataset(ds['validation'], model)

    argument_kwargs = {}
    if args.push_to_hub:
        assert args.hub_model_id is not None
        argument_kwargs['push_to_hub'] = True,
        argument_kwargs['hub_model_id'] = args.hub_model_id

    model.fit(
        train_ds=train_ds,
        valid_ds=valid_ds,
        output_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_kwargs={
            'w1': args.w1,
            'w2': args.w2,
            'w3': args.w3,
            'cosine_tau': args.cosine_tau,
            'ibn_tau': args.ibn_tau,
            'angle_tau': args.angle_tau,
        },
        fp16=args.fp16,
        argument_kwargs=argument_kwargs
    )

    return model


def evaluate(args):
    args.model_name_or_path = args.pretrained_model_path
    args.angle_name_or_path = f"${args.save_dir}/best-checkpoint"

    model = EmbModel(model_name_or_path=args.model_name_or_path,
                     angle_name_or_path=args.angle_name_or_path,
                     normalize_embeddings=False,  # normlize embedding will harm the performance of classification task
                     query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                     pooling_method=args.pooling_strategy,
                     batch_size=args.batch_size)

    # tasks = mteb.get_tasks(task_types=[args.task_type], languages=['eng'])
    # task_names = [t.metadata.name for t in tasks]
    task_names = [
        'ArguAna',
        'ClimateFEVER',
        'CQADupstackRetrieval',
        'DBPedia',
        'FEVER',
        'FiQA2018',
        'HotpotQA',
        'MSMARCO',
        'NFCorpus',
        'NQ',
        'QuoraRetrieval',
        'SCIDOCS',
        'SciFact',
        'Touche2020',
        'TRECCOVID',
    ]
    random.shuffle(task_names)

    for task in task_names:
        if task in ['MSMARCOv2']:
            print('Skip task: {}, since it has no test split'.format(task))
            continue

        # if 'CQADupstack' in task or task in ['Touche2020', 'SciFact', 'TRECCOVID', 'NQ',
        #                                      'NFCorpus', 'MSMARCO', 'HotpotQA', 'FiQA2018',
        #                                      'FEVER', 'DBPedia', 'ClimateFEVER', 'SCIDOCS', ]:
        #     instruction = RETRIEVAL_INSTRUCT
        # else:
        #     instruction = None
        #
        # model.query_instruction_for_retrieval = instruction

        print(f"Start task: {task}")
        evaluation = MTEB(
            tasks=[task],
            task_langs=["eng"],
            eval_splits=["test" if task not in ["MSMARCO"] else "dev"],
        )
        evaluation.run(
            model,
            output_folder=f"{args.output_dir}/mteb_results/en/{args.angle_name_or_path.split('/')[-1]}",
        )
        print(f"Finished task: {task}")


def prepare_contrastive_dataset(dataset, model):
    # Function to create positive pairs
    def create_positive_pair(example):
        return {
            'text1': example['query'],
            'text2': example['answer'],
            'label': 1  # Positive pair label
        }

    # Function to create negative pairs by shuffling answers
    def create_negative_pair(example, dataset):
        # Pick a random answer that isn't the correct one
        ind = random.randint(0, len(dataset) - 1)
        random_answer = dataset[ind]['answer']
        while random_answer == example['answer']:
            ind = random.randint(0, len(dataset) - 1)
            random_answer = dataset[ind]['answer']
        return {
            'text1': example['query'],
            'text2': random_answer,
            'label': 0  # Negative pair label
        }

    positive_pairs = dataset.map(create_positive_pair)

    negative_pairs = dataset.map(lambda x: create_negative_pair(x, dataset))

    combined_ds = concatenate_datasets([positive_pairs, negative_pairs])

    return combined_ds.shuffle(args.dataset_seed).map(
        AngleDataTokenizer(
            model.tokenizer, model.max_length, prompt_template=args.prompt
        ),
        num_proc=args.workers,
    )


if __name__ == '__main__':
    args = parse_arguments()

    print('Start training...')
    train(args)
    print('Training finished.')

    if args.run_eval:
        print('Start evaluation...')
        evaluate(args)
        print('Evaluation finished.')
