import torch, os, typing
import numpy as np

def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Given a 1-D dataset, return a batch of inputs and targets.
    Inputs are the first context_length tokens, and targets are the next context_length tokens.
    """
    # Randomly sample batch_size starting indices for the sequences
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    # Create input and target tensors
    inputs = np.array([dataset[i:i + context_length] for i in start_indices], dtype=np.int64)
    targets = np.array([dataset[i + 1:i + context_length + 1] for i in start_indices], dtype=np.int64)

    # Convert to torch tensors and move to the specified device
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    """
    Save the model and optimizer state to a checkpoint file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    Load the model and optimizer state from a checkpoint file.
    Returns the iteration number to resume training from.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']