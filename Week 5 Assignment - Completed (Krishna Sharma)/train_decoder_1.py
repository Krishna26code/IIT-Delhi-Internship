import argparse
import os
import torch
import wandb
import ast

from accelerate import Accelerator
from data.processed import ItemData, RecDataset, SeqData
from modules.semids import batch_to, next_batch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.inv_sqrt import InverseSquareRootScheduler
from modules.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def train(iterations=500000,
          batch_size=64,
          learning_rate=0.001,
          weight_decay=0.01,
          dataset_folder="dataset/amazon",
          save_dir_root="out/",
          dataset=RecDataset.AMAZON,
          pretrained_rqvae_path=None,
          pretrained_decoder_path=None,
          split_batches=True,
          amp=False,
          wandb_logging=False,
          force_dataset_process=False,
          mixed_precision_type="fp16",
          gradient_accumulate_every=1,
          save_model_every=1000000,
          partial_eval_every=1000,
          full_eval_every=10000,
          vae_input_dim=768,
          vae_embed_dim=16,
          vae_hidden_dims=[18,18],
          vae_codebook_size=32,
          vae_codebook_normalize=False,
          vae_n_cat_feats=0,
          vae_n_layers=3,
          dataset_split="beauty",
          push_vae_to_hf=False,
          train_data_subsample=True,
          vae_hf_model_name="rqvae-amazon-beauty",
          max_grad_norm=None,
          t5_d_model=128,
          t5_num_heads=6,
          t5_d_ff=1024,
          t5_num_layers=4,
          top_k_for_generation=10,
          should_add_sep_token=True,
          num_user_bins=None,
          top_k_eval_list=[1, 5, 10]):
    
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported:{dataset}")

    if pretrained_rqvae_path is None:
        print("=" * 70)
        print("WARNING: pretrained_rqvae_path is None.")
        print("The RQ-VAE inside the tokenizer will use RANDOM, UNTRAINED weights.")
        print("Semantic IDs will be meaningless and the decoder will learn nothing")
        print("useful (loss may collapse to ~0 due to codebook collapse).")
        print("Pass --pretrained_rqvae_path out/checkpoint_<N>.pt from a completed")
        print("train_rqvae.py run before training the decoder for real results.")
        print("=" * 70)
    
    if wandb_logging:
        params = locals()
    
    accelerator = Accelerator(split_batches=split_batches,
                              mixed_precision=mixed_precision_type if amp else "no")
    
    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(project="gen-retrieval-decoder-training", config=params)
    
    item_dataset  = ItemData(root=dataset_folder,
                            dataset=dataset,
                            force_process=force_dataset_process,
                            split=dataset_split)
    train_dataset = SeqData(root=dataset_folder,
                            dataset=dataset,
                            is_train=True,
                            subsample=train_data_subsample,
                            split=dataset_split)
    eval_dataset  = SeqData(root=dataset_folder,
                            dataset=dataset,
                            is_train=False,
                            subsample=False,
                            split=dataset_split)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = cycle(train_dataloader)
    eval_dataloader  = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    tokenizer = SemanticIdTokenizer(input_dim=vae_input_dim,
                                    output_dim=vae_embed_dim,
                                    hidden_dims=vae_hidden_dims,
                                    codebook_size=vae_codebook_size,
                                    n_layers=vae_n_layers,
                                    n_cat_feats=vae_n_cat_feats,
                                    rqvae_weights_path=pretrained_rqvae_path,
                                    rqvae_codebook_normalize=vae_codebook_normalize)
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)
    
    codebooks = tokenizer.cached_ids[:, :vae_n_layers].cpu()

    model = EncoderDecoderRetrievalModel(codebooks=codebooks,
                                         num_hierarchies=vae_n_layers,
                                         num_embeddings_per_hierarchy=vae_codebook_size,
                                         t5_d_model=t5_d_model,
                                         t5_num_heads=t5_num_heads,
                                         t5_d_ff=t5_d_ff,
                                         t5_num_layers=t5_num_layers,
                                         top_k_for_generataion=top_k_for_generation,
                                         should_add_sep_token=should_add_sep_token,
                                         num_user_bins=num_user_bins)
    model = torch.compile(model)

    optimizer = AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=10000)

    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1
    
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    metrics_accumulator = TopKAccumulator(ks=top_k_eval_list)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")

    with tqdm(initial=start_iter, 
              total=start_iter+iterations, disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            train_debug_metrics = {}

            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every
                
                total_loss += loss.detach().cpu()

                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data)

                accelerator.backward(loss)

                assert model.item_sid_embedding_table.weight.grad is not None

                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                if max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

                accelerator.wait_for_everyone()

                if (iter + 1) % partial_eval_every == 0:
                    model.eval()
                    eval_loss = 0.0
                    for batch in eval_dataloader:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)
                        with torch.no_grad():
                            eval_loss = model(tokenized_data).loss.item()
                    
                    if wandb_logging and accelerator.is_main_process:
                        wandb.log({"eval_loss": eval_loss})
                
                if (iter + 1) % full_eval_every == 0:
                    model.eval()
                    with tqdm(eval_dataloader, desc=f"Eval {iter + 1}",
                              disable=not accelerator.is_main_process) as pbar_eval:
                        for batch in pbar_eval:
                            data = batch_to(batch, device)
                            tokenized_data = tokenizer(data)

                            with torch.no_grad():
                                generated = model.generate_next_sem_id(
                                    tokenized_data, top_k=True, temperature=1
                                )
                            
                            actual = tokenized_data.sem_ids_fut[:, :vae_n_layers]
                            metrics_accumulator.accumulate(actual=actual, top_k=generated.sem_ids)
                    
                    eval_metrics = metrics_accumulator.reduce()
                    print(eval_metrics)
                    if accelerator.is_main_process and wandb_logging:
                        wandb.log(eval_metrics)
                    metrics_accumulator.reset()

                if accelerator.is_main_process:
                    if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                        state = {"iter":iter,
                                 "model":model.state_dict(),
                                 "optimizer":optimizer.state_dict(),
                                 "scheduler":lr_scheduler.state_dict()}
                        
                        if not os.path.exists(save_dir_root):
                            os.makedirs(save_dir_root)
                        
                        torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")
                    
                    if wandb_logging:
                        wandb.log(
                            {
                                "learning_rate":optimizer.param_groups[0]["lr"],
                                "total_loss":total_loss,
                                **train_debug_metrics
                            }
                        )
                
                pbar.update(1)
        
    if wandb_logging:
        wandb.finish()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dataset_folder", default="dataset/amazon")
    parser.add_argument("--dataset", default="AMAZON")
    parser.add_argument("--pretrained_rqvae_path", default=None)
    parser.add_argument("--pretrained_decoder_path", default=None)
    parser.add_argument("--save_dir_root", default="out/")
    parser.add_argument("--split_batches", type=bool, default=True)
    parser.add_argument("--amp", type=bool,default=False)
    parser.add_argument("--wandb_logging", type=bool, default=False)
    parser.add_argument("--force_dataset_process", type=bool, default=False)
    parser.add_argument("--mixed_precision_type", default="fp16")
    parser.add_argument("--gradient_accumulate_every", type=int, default=1)
    parser.add_argument("--save_model_every", type=int, default=1000000)
    parser.add_argument("--partial_eval_every", type=int, default=1000)
    parser.add_argument("--full_eval_every", type=int, default=10000)
    parser.add_argument("--vae_n_cat_feats", type=int, default=0)
    parser.add_argument("--vae_input_dim", type=int, default=768)
    parser.add_argument("--vae_embed_dim", type=int, default=16)
    parser.add_argument("--vae_hidden_dims", default="[18,18]")
    parser.add_argument("--vae_codebook_size", type=int, default=32)
    parser.add_argument("--vae_codebook_normalize", type=bool, default=False)
    parser.add_argument("--vae_n_layers", type=int, default=3)
    parser.add_argument("--dataset_split", default="beauty")
    parser.add_argument("--push_vae_to_hf", type=bool, default=False)
    parser.add_argument("--train_data_subsample", type=bool, default=True)
    parser.add_argument("--vae_hf_model_name", default="rqvae-amazon-beauty")
    parser.add_argument("--max_grad_norm", default=None)
    parser.add_argument("--t5_d_model", type=int, default=128)
    parser.add_argument("--t5_num_heads", type=int, default=6)
    parser.add_argument("--t5_d_ff", type=int, default=1024)
    parser.add_argument("--t5_num_layers", type=int, default=4)
    parser.add_argument("--top_k_for_generation", type=int, default=10)
    parser.add_argument("--should_add_sep_token", type=bool, default=True)
    parser.add_argument("--num_user_bins", default=None)
    parser.add_argument("--top_k_eval_list", default="[1, 5, 10]")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    args.dataset = RecDataset[args.dataset]
    args.vae_hidden_dims = ast.literal_eval(args.vae_hidden_dims)
    args.top_k_eval_list = ast.literal_eval(args.top_k_eval_list)
    train(args.iterations, args.batch_size, args.learning_rate, args.weight_decay, args.dataset_folder,
          args.save_dir_root,args.dataset, args.pretrained_rqvae_path, args.pretrained_decoder_path,
          args.split_batches, args.amp, args.wandb_logging, args.force_dataset_process, args.mixed_precision_type,
          args.gradient_accumulate_every, args.save_model_every, args.partial_eval_every, args.full_eval_every,
          args.vae_input_dim, args.vae_embed_dim, args.vae_hidden_dims, args.vae_codebook_size,
          args.vae_codebook_normalize, args.vae_n_cat_feats, args.vae_n_layers, args.dataset_split, args.push_vae_to_hf,
          args.train_data_subsample, args.vae_hf_model_name, args.max_grad_norm, args.t5_d_model, args.t5_num_heads,
          args.t5_d_ff, args.t5_num_layers, args.top_k_for_generation, args.should_add_sep_token, args.num_user_bins,
          args.top_k_eval_list)