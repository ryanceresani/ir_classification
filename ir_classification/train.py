from collections import Counter
from logging import log
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def predict(model: nn.Module, text: torch.Tensor) -> int:
    """Predicts the class of a specific text given converted features.
    
    Args:
        model: the model to use for prediction/inferrence
        text: the previously converted text (using the prior dictionary)
    
    Returns:
        The predicted label for the provided text.
    """
    model.eval()
    no_offset = torch.tensor([0])
    with torch.no_grad():
        pred_scores = model(text, no_offset)
        pred_label = pred_scores.argmax(1).item()
        return pred_label


def train_epoch(
    epoch_num: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
    dataloader: data.DataLoader,
    start_iter: int = 0,
    log_interval: int = 100,
    writer: SummaryWriter = None,
) -> int:
    """Performs training on a single pass through a dataloader.
    
    Args:
        epoch_num: The current number of this epoch of training.    
        model: The PyTorch module to train
        optimizer: The optimizer to use for training.
        loss_function: The function that calculates loss between truth and prediction.
        dataloader: Provides a properly formatted batch of data at each iteration.
        start_iter: The iteration this epoch started on. Used for plotting.
        log_interval: How often to log the scores.
        writier: Tensorboard summary writer. 
    
    Returns:
        The value of the start_iter plus number of batches performed this epoch.
    """
    batch_counter = start_iter
    model.train()
    with tqdm(dataloader, unit=" batch", bar_format="{desc:>20}{percentage:3.0f}%|{bar}{r_bar}") as tepoch:
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
    """Performs a single training step on a model.

    Args:
        batch: A previously formatted batch of data.
        model: Torch model to perform training on.
        optimizer: The optmizer class used in training
        loss_function: The callable function to generate loss between prediction and truth.

    Returns:
        The different metrics generated this training step.
    """
    labels, text, offsets, *_ = batch

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
        predicted_labels.cpu().numpy(),
        average="macro",
        zero_division=0,
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
    writer: SummaryWriter = None,
)-> Dict[str, float]:
    """Performs validation on a single pass through a dataloader.
    
    Args:
        epoch_num: The current number of this epoch of training.    
        model: The PyTorch module to train
        loss_function: The function that calculates loss between truth and prediction.
        dataloader: Provides a properly formatted batch of data at each iteration.
        writier: Tensorboard summary writer. 
    
    Returns:
        The average validation metrics for the whole dataset.
    """
    model.eval()
    aggregate_results = Counter()
    with tqdm(dataloader, unit=" batch", bar_format="{desc:>20}{percentage:3.0f}%|{bar}{r_bar}") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Validation: {epoch_num}")
            results = evaluate_step(batch, model, loss_function)
            tepoch.set_postfix(loss=results["loss"], accurracy=results["accuracy"])
            aggregate_results += Counter(results)

        average_results = {
            key: aggregate_results[key] / tepoch.total for key in aggregate_results
        }

    writer.add_scalars("validation", average_results, epoch_num)
    return average_results


def evaluate_step(
    batch: Tuple[torch.Tensor, ...],
    model: nn.Module,
    loss_function: Callable,
) -> Dict[str, float]:
    """Performs a single validation step on a model.

    Args:
        batch: A previously formatted batch of data.
        model: Torch model to perform training on.
        loss_function: The callable function to generate loss between prediction and truth.

    Returns:
        The different metrics generated this training step.
    """

    labels, text, offsets, *_ = batch

    with torch.no_grad():
        predicted_scores = model(text, offsets)
        loss = loss_function(predicted_scores, labels)
        predicted_labels = predicted_scores.argmax(1)
        accuracy = (predicted_labels == labels).sum().item() / labels.size(0)
        precision, recall, fscore, support = precision_recall_fscore_support(
            labels.cpu().numpy(),
            predicted_labels.cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    results = {
        "loss": loss.detach().cpu().item(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }
    return results
