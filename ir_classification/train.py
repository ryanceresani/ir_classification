from typing import Any, Callable, Dict, Tuple
import torch
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter


def train_epoch(
    epoch_num: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
    dataloader: data.DataLoader,
    log_interval: int = 500,
    tensorboard: SummaryWriter = None
):
    model.train()
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch_num}")
            results = train_step(batch, model, optimizer, loss_function)
            tepoch.set_postfix(loss=results["loss"], accurracy=results["accuracy"])


def train_step(
    batch: Tuple[torch.Tensor, ...],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
) -> Dict[str, float]:
    label, text, offsets = batch

    optimizer.zero_grad()
    predicted_scores = model(text, offsets)
    loss = loss_function(predicted_scores, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()

    predicted_labels = predicted_scores.argmax(1)
    accuracy = (predicted_labels == label).sum().item() / label.size(0)
    
    precision, recall, fscore, support = precision_recall_fscore_support(
        label.detach().cpu().numpy(),
        predicted_labels,
        average="weighted",
        labels=np.unique(predicted_labels)
    )

    results = {
        "loss": loss.detach().cpu().numpy(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }
    return results

def evaluate():
    pass
