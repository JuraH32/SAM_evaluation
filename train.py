import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import wandb

from sam import SAM
from util.bypass_bn import enable_running_stats, disable_running_stats


# Enum for optimizers
class OptimizerType:
    SGD = 'SGD'
    ADAM = 'ADAM'
    SAM = 'SAM'

def get_optimizer(model, optimizer):
    optimizer_type = optimizer.get('optimizer_type', OptimizerType.SGD)
    momentum = optimizer.get('momentum', 0.9)
    weight_decay = optimizer.get('weight_decay', 5e-4)
    adaptive = optimizer.get('adaptive', False)
    rho = optimizer.get('rho', 0.05)
    learning_rate = optimizer.get('learning_rate', 0.001)

    if optimizer_type == OptimizerType.ADAM:
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    base_optimizer = optim.SGD

    if optimizer_type == OptimizerType.SGD:
        return base_optimizer(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


    return SAM(model.parameters(), base_optimizer, rho=rho, adaptive=adaptive,
                        lr=learning_rate, momentum=momentum, weight_decay=weight_decay)



def train_multiple_models(configs, train_loader, val_loader, dataset_name, device, use_wandb=True):
    """
    Train multiple models with different configurations.
    Args:
        configs (list): List of dictionaries containing model configurations. Each
                        dictionary should contain some of these values 'model', 'criterion', 'optimizer', 'scheduler',
                        'num_epochs' 'save_dir', 'early_stopping_patience' and 'model_name'.
    """
    for config in configs:
        model = config['model']
        criterion = config.get('criterion', nn.MSELoss)
        optimizer = config.get('optimizer', {'optimizer_type': OptimizerType.SGD})
        num_epochs = config.get('num_epochs', 100)
        save_dir = config.get('save_dir', 'checkpoints')
        early_stopping_patience = config.get('early_stopping_patience', 10)
        model_name = config.get('model_name', 'model')

        run = wandb.init(project=config['model_name'], name=config['model_name'], config={
            "architecture": config['model_name'],
            "dataset": dataset_name,
            "learning_rate": config['optimizer'].param_groups[0]['lr'],
            "batch_size": config['batch_size'],
            "epochs": num_epochs,
            "optimizer": config['optimizer'].__class__.__name__,
        })

        print(f'\nTraining model: {model_name}')
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,

            num_epochs=num_epochs,
            device=device,
            save_dir=save_dir,
            early_stopping_patience=early_stopping_patience,
            model_name=model_name,
            use_wandb=use_wandb
        )



def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=25,
        device='cuda',
        save_dir='checkpoints',
        early_stopping_patience=10,
        model_name='model',
        use_wandb=True
):
    """
    Train a PyTorch model.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train
        device: Device to train on ('cuda', 'cpu', 'mps')
        save_dir: Directory to save model checkpoints
        early_stopping_patience: Number of epochs to wait before early stopping
        model_name: Name for saving the model
        use_wandb: Whether to log metrics to Weights & Biases

    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    # Create directory for saving checkpoints if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Initialize variables for tracking best model and early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Get current time for logging
    start_time = time.time()

    # Move model to device
    model.to(device)

    optimizer = get_optimizer(model, optimizer)
    optimizer_type = optimizer.get('optimizer_type', OptimizerType.SGD)

    # Main training loop
    for epoch in range(num_epochs):
        print(f'\n{"=" * 50}')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'{"=" * 50}')

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Training')

        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if optimizer_type == OptimizerType.SAM:
                # SAM optimization steps
                enable_running_stats(model)
                predictions = model(inputs)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # Disable running stats for the second step
                # This is important for models with BatchNorm layers
                disable_running_stats(model)
                predictions = model(inputs)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)

            else:
                predictions = model(inputs)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


            with torch.no_grad():
                # Get predictions
                _, preds = torch.max(predictions, 1)

                # Update running statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                running_loss += batch_loss
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar with more detailed metrics
                train_pbar.set_postfix({
                    'batch_loss': f'{batch_loss / inputs.size(0):.4f}',
                    'batch_acc': f'{batch_acc:.4f}',
                    'avg_loss': f'{running_loss / ((batch_idx + 1) * inputs.size(0)):.4f}'
                })

                if use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        "batch": epoch * len(train_loader) + batch_idx,
                        "batch_loss": batch_loss / inputs.size(0),
                        "batch_acc": batch_acc.item()
                    })


        # Calculate epoch statistics
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        # No gradient computation for validation
        with torch.no_grad():
            # Progress bar for validation batches
            val_pbar = tqdm(val_loader, desc=f'Validation')

            for batch_idx, (inputs, labels) in enumerate(val_pbar):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Get predictions
                _, preds = torch.max(outputs, 1)

                # Update running statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                running_loss += batch_loss
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar with more detailed metrics
                val_pbar.set_postfix({
                    'batch_loss': f'{batch_loss / inputs.size(0):.4f}',
                    'batch_acc': f'{batch_acc:.4f}',
                    'avg_loss': f'{running_loss / ((batch_idx + 1) * inputs.size(0)):.4f}'
                })

        # Calculate epoch statistics
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch results with better formatting
        print(f'\nEpoch {epoch + 1} Results:')
        print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        # Record history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_acc'].append(epoch_val_acc.item())

        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc.item(),
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc.item(),
                "learning_rate": current_lr
            })

        # Check if this is the best model so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save best model
            best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
            }, best_model_path)

            if use_wandb:
                wandb.save(best_model_path)  # Save best model to wandb

            # Reset early stopping counter
            early_stopping_counter = 0
            print(f'New best model saved with validation loss: {epoch_val_loss:.4f}')
        else:
            early_stopping_counter += 1
            print(f'Early stopping counter: {early_stopping_counter}/{early_stopping_patience}')

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc,
        }, checkpoint_path)

        # Check for early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    # Calculate and print training time
    time_elapsed = time.time() - start_time
    print(
        f'\nTraining completed in {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Load best model weights
    best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(
            f'Loaded best model from epoch {checkpoint["epoch"] + 1} with validation loss: {checkpoint["val_loss"]:.4f}')

    # Plot training curves
    fig = plot_training_curves(history, save_dir, model_name)

    # Log the figure to wandb
    if use_wandb:
        wandb.log({"training_curves": wandb.Image(fig)})

    return model, history


def plot_training_curves(history, save_dir, model_name):
    """
    Plot training and validation curves

    Returns:
        fig: Figure object for the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_curves.png'))

    # Don't close the figure so we can return it
    return fig


def main():
    """
    Example usage of train_model function with wandb integration
    """
    # This is a placeholder - you would need to:
    # 1. Create/load your model
    # 2. Define your datasets and data loaders
    # 3. Choose loss function and optimizer
    # 4. Call train_model with your configurations

    # Example (commented out):
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb
    run = wandb.init(
        project="your-project-name",
        name="experiment-name",
        config={
            "architecture": "YourModel",
            "dataset": "YourDataset",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "optimizer": "Adam",
            # Add any other hyperparameters you want to track
        }
    )

    # Create model
    model = YourModel()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Train the model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device,
        save_dir='checkpoints',
        early_stopping_patience=10,
        model_name='my_model',
        use_wandb=True
    )

    # Finish wandb run
    wandb.finish()
    """
    pass


if __name__ == "__main__":
    main()
