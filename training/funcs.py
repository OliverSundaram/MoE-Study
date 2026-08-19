import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import time
from pathlib import Path

from models.modules import MoE, LLM

from transformers import GPT2Tokenizer



RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

def train(model: LLM,
          tokenizer,
          loss_fn: nn.CrossEntropyLoss,
          optimizer: torch.optim.AdamW,
          scheduler,
          train_loader: DataLoader,
          val_loader: DataLoader,
          test_loader: DataLoader,
          device: torch.device,
          checkpoint_name: str,
          val_every_steps: int = 100,
          max_norm: float = 1.0,
          grad_accum_steps: int = 1,
          use_amp: bool = True,):

    checkpoint_dir = RUNS_DIR / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    train_losses_with_aux, train_losses_no_aux, val_losses_with_aux, val_losses_no_aux = [], [], [], []
    steps, val_steps, train_step_times  = [], [], []
    step = 0

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    model.train()
    optimizer.zero_grad()

    is_moe = model.config.is_moe

    for X, y in tqdm(train_loader, desc="Training"):
        step_start = time.time()
        step += 1
        steps.append(step)

        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            logits_train = model(X).logits
            loss_train = loss_fn(logits_train.view(-1, logits_train.shape[-1]), y.view(-1))
            train_losses_no_aux.append(loss_train.item())

            if is_moe:
                total_aux_loss = sum(m.aux_loss for m in model.modules() if isinstance(m, MoE))
                loss_train += total_aux_loss
            train_losses_with_aux.append(loss_train.item())

            loss = loss_train / grad_accum_steps

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        if step % val_every_steps == 0:
            val_loss_with_aux_total = 0.0
            val_loss_no_aux_total = 0.0
            model.eval()
            val_steps.append(step)

            with torch.inference_mode():
                for X_val, y_val in tqdm(val_loader, desc="Validation"):
                    X_val, y_val = X_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)

                    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                        logits_val = model(X_val).logits
                        loss_val = loss_fn(logits_val.view(-1, logits_val.shape[-1]), y_val.view(-1))
                        val_losses_no_aux.append(loss_val.item())
                        val_loss_no_aux_total += loss_val.item()

                        if is_moe:
                            total_aux_loss = sum(m.aux_loss for m in model.modules() if isinstance(m, MoE))
                            loss_val += total_aux_loss
                        val_losses_with_aux.append(loss_val.item())
                        val_loss_with_aux_total += loss_val.item()

                val_loss_no_aux_avg = val_loss_no_aux_total / len(val_loader)
                val_loss_with_aux_avg = val_loss_with_aux_total / len(val_loader)

            step_dir = checkpoint_dir / f"step_{step}"
            model.save_pretrained(step_dir)
            tokenizer.save_pretrained(step_dir)
            torch.save(
                {
                    "step": step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,

                    "train_losses_with_aux": train_losses_with_aux,
                    "train_losses_no_aux": train_losses_no_aux,
                    "val_losses_with_aux": val_losses_with_aux,
                    "val_losses_no_aux": val_losses_no_aux,
                    "steps": steps,
                    "val_steps": val_steps,
                    "train_step_times": train_step_times
                },
                step_dir / "trainer_state.pt"
            )

            print(f"\n[Step {step}] Train Loss (no aux): {train_losses_no_aux[-1]:.4f} "
                  f"| Val Loss (no aux): {val_loss_no_aux_avg:.4f} "
                  f"| Val Loss (with aux): {val_loss_with_aux_avg:.4f}")
            model.train()

        train_step_times.append(round(time.time() - step_start, 4))

    if step % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # ----------
    # Test model
    # ----------
    print("\nTraining complete. Running full test evaluation...\n")
    test_losses_with_aux, test_losses_no_aux = [], []
    test_loss_with_aux_total = 0.0
    test_loss_no_aux_total = 0.0
    model.eval()

    with torch.inference_mode():
        for X_test, y_test in tqdm(test_loader, desc="Testing"):
            X_test, y_test = X_test.to(device, non_blocking=True), y_test.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits_test = model(X_test).logits
                loss_test = loss_fn(logits_test.view(-1, logits_test.shape[-1]), y_test.view(-1))
                test_losses_no_aux.append(loss_test.item())
                test_loss_no_aux_total += loss_test.item()

                if is_moe:
                    total_aux_loss = sum(m.aux_loss for m in model.modules() if isinstance(m, MoE))
                    loss_test += total_aux_loss
                test_losses_with_aux.append(loss_test.item())
                test_loss_with_aux_total += loss_test.item()

        test_loss_no_aux_avg = test_loss_no_aux_total / len(test_loader)
        test_loss_with_aux_avg = test_loss_with_aux_total / len(test_loader)

    training_time = time.time() - start
    print(f"Test Loss (no aux): {test_loss_no_aux_avg:.4f} "
          f"| Test Loss (with aux): {test_loss_with_aux_avg:.4f} "
          f"| Total time: {training_time:.4f}s")

    # ----------
    # Save final model
    # ----------
    final_dir = checkpoint_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save(
            {
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,

                "train_losses_with_aux": train_losses_with_aux,
                "train_losses_no_aux": train_losses_no_aux,
                "val_losses_with_aux": val_losses_with_aux,
                "val_losses_no_aux": val_losses_no_aux,
                "test_losses_with_aux": test_losses_with_aux,
                "test_losses_no_aux": test_losses_no_aux,
                "test_loss_with_aux_avg": test_loss_with_aux_avg,
                "test_loss_no_aux_avg": test_loss_no_aux_avg,
                "steps": steps,
                "val_steps": val_steps,
                "train_step_times": train_step_times,
            },
            final_dir / "final_state.pt"
        )

    return (
        model,
        train_losses_with_aux, train_losses_no_aux,
        val_losses_with_aux, val_losses_no_aux,
        test_losses_with_aux, test_losses_no_aux,
        val_steps,
        training_time,
        train_step_times,
    )