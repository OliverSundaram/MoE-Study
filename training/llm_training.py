import torch
from torch import nn
from torch.utils.data import DataLoader

from training.funcs import train

from models.modules import LLM, LLMConfig

from transformers import AutoTokenizer

from pathlib import Path

from data_preparation.dataset import LLMDataset



def main():
    # --------------------------------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------------------------------
    DENSE_MODEL = False

    BATCH_SIZE = 2
    LR_RATE = 3e-4
    WEIGHT_DECAY = 0.1
    MAX_NORM = 1.0
    GRAD_ACCUM_STEPS = 4
    VAL_EVERY_STEPS = 7000
    NUM_WORKERS = 2
    CONFIG = LLMConfig() if DENSE_MODEL else LLMConfig(hidden_dim=1536, is_moe=True, n_experts=4, top_k=2)

    # --------------------------------------------------------------------------------
    # Device Selection & Seed
    # --------------------------------------------------------------------------------
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required for this training run, but none was detected.")
    print(f"Using device: {device}")

    # Optimize matrix multiplication on Tensor Cores is available
    torch.set_float32_matmul_precision("high")

    # --------------------------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------------------------
    print("Loading Datasets...")

    DATA_DIR = Path(__file__).resolve().parent.parent / "data_preparation" / "data"
    train_ds = torch.load(DATA_DIR / "train_dataset.pt", weights_only=False)
    val_ds = torch.load(DATA_DIR / "val_dataset.pt", weights_only=False)
    test_ds = torch.load(DATA_DIR / "test_dataset.pt", weights_only=False)

    train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=False)
    test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=False)

    # --------------------------------------------------------------------------------
    # Model Initialization
    # --------------------------------------------------------------------------------
    model = LLM(CONFIG).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # --------------------------------------------------------------------------------
    # Loss, Optimizer, & Scheduler
    # --------------------------------------------------------------------------------
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=LR_RATE)

    loss_fn = nn.CrossEntropyLoss()

    total_optim_steps = max(1, len(train_loader) // GRAD_ACCUM_STEPS)
    warmup_steps = max(2, int(0.03 * total_optim_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR_RATE,
        total_steps=total_optim_steps,
        pct_start=warmup_steps / total_optim_steps,
        anneal_strategy="cos",
    )
    # --------------------------------------------------------------------------------
    # Run Training Loop
    # --------------------------------------------------------------------------------
    print("Starting training...")

    (model,
     train_losses_with_aux, train_losses_no_aux,
     val_losses_with_aux, val_losses_no_aux,
     test_losses_with_aux, test_losses_no_aux,
     val_steps,
     training_time,
     train_step_times) = train(
        model=model,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_name="dense_checkpoint" if DENSE_MODEL else "moe_checkpoint",
        val_every_steps=VAL_EVERY_STEPS,
        max_norm=MAX_NORM,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        use_amp=True,
    )

if __name__ == "__main__":
    main()