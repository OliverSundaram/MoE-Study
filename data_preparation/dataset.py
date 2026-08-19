from pathlib import Path

from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent / "data"



class LLMDataset(Dataset):

    def __init__(self, ids: torch.Tensor, context_size: int, stride: int):

        if not isinstance(ids, torch.Tensor):
            self.ids = torch.tensor(ids, dtype=torch.long)
        else:
            self.ids = ids.long()

        self.context_size = context_size
        self.stride = stride

        max_start_idx = len(self.ids) - self.context_size
        self.start_indices = list(range(0, max_start_idx, self.stride))

    def __getitem__(self, idx: int):

        start_idx = self.start_indices[idx]
        inputs = self.ids[start_idx : start_idx + self.context_size]
        targets = self.ids[start_idx + 1 : start_idx + self.context_size + 1]
        return inputs, targets

    def __len__(self):
        return len(self.start_indices)


def get_dataset(split: str,
                context_size: int = 1024,
                stride: int = 1024) -> LLMDataset:

    splits = ["train", "val", "test"]
    split = split.strip().lower()
    if split not in splits:
        raise ValueError(f"Split {split} is not valid. ({splits})")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    with open(DATA_DIR / f"{split}_data", "r", encoding="utf-8") as f:
        raw_data = f.read()

    token_ids = tokenizer(raw_data)["input_ids"]
    ids_tensor = torch.tensor(token_ids, dtype=torch.long)

    return LLMDataset(ids_tensor, context_size=context_size, stride=stride)

if __name__ == "__main__":

    # train_ds = get_dataset("train", 1024, 1024)
    # torch.save(train_ds, "data/train_dataset.pt")

    # val_ds = get_dataset("val", 1024, 1024)
    # torch.save(val_ds, "data/val_dataset.pt")

    # test_ds = get_dataset("test", 1024, 1024)
    # torch.save(test_ds, "data/test_dataset.pt")

    pass