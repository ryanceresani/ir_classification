from collections import Counter
from logging import log
from typing import Any, Callable, Dict, Tuple
import torch
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import ipdb
from torch.utils.tensorboard import SummaryWriter


def train_epoch(
    epoch_num: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
    dataloader: data.DataLoader,
    start_iter: int = 0,
    log_interval: int = 100,
    writer: SummaryWriter = None
):
    batch_counter = start_iter
    model.train()
    with tqdm(dataloader, unit=" batch") as tepoch:
        for batch in tepoch:
            batch_counter += 1
            tepoch.set_description(f"Epoch {epoch_num}")
            results = train_step(batch, model, optimizer, loss_function)
            tepoch.set_postfix(loss=results["loss"], accurracy=results["accuracy"])

            if writer is not None and batch_counter % log_interval == 0:
                writer.add_scalars("training", results, batch_counter)

    return batch_counter

def train_step(
    batch: Tuple[torch.Tensor, ...],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
) -> Dict[str, float]:
    labels, text, offsets = batch

    optimizer.zero_grad()
    predicted_scores = model(text, offsets)
    loss = loss_function(predicted_scores, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()

    predicted_labels = predicted_scores.argmax(1)

    accuracy = (predicted_labels == labels).sum().item() / labels.size(0)
    
    precision, recall, fscore, support = precision_recall_fscore_support(
        labels.detach().cpu().numpy(),
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    results = {
        "loss": loss.detach().cpu().item(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }
    return results

def evaluate_epoch(
    epoch_num: int,
    model: nn.Module,
    loss_function: Callable,
    dataloader: data.DataLoader,
    writer: SummaryWriter = None
):
    model.eval()
    aggregate_results = Counter()
    with tqdm(dataloader, unit=" batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Validation: {epoch_num}")
            results = evaluate_step(batch, model, loss_function)
            tepoch.set_postfix(loss=results["loss"], accurracy=results["accuracy"])
            aggregate_results += Counter(results)
        
        average_results = {key: aggregate_results[key] / tepoch.total for key in aggregate_results}
    
    writer.add_scalars("validation", average_results, epoch_num)
    return average_results

def evaluate_step(
    batch: Tuple[torch.Tensor, ...],
    model: nn.Module,
    loss_function: Callable,
) -> Dict[str, float]:
    labels, text, offsets = batch

    with torch.no_grad():
        predicted_scores = model(text, offsets)
        loss = loss_function(predicted_scores, labels)
        predicted_labels = predicted_scores.argmax(1)
        accuracy = (predicted_labels == labels).sum().item() / labels.size(0)
        precision, recall, fscore, support = precision_recall_fscore_support(
            labels.detach().cpu().numpy(),
            predicted_labels,
            average="weighted",
            zero_division=0
        )
    results = {
        "loss": loss.detach().cpu().item(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }
    return results
