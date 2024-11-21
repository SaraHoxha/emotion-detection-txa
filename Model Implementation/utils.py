import torch
from torch import optim, nn
from sklearn.model_selection import train_test_split, KFold
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

# Training and validation function
def train_and_validate(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    """
    Trains and validates a PyTorch model over multiple epochs.

    Args:
        model (torch.nn.Module): The model to train and validate.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's weights.
        criterion (torch.nn.Module): The loss function used to compute the error.
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of training data (inputs and labels).
        val_loader (torch.utils.data.DataLoader): DataLoader providing batches of validation data (inputs and labels).
        epochs (int): The number of epochs to train the model.
        device (torch.device): The device (CPU or GPU) to run the computations on.

    Returns:
        tuple: A tuple containing three lists:
            - train_losses (list[float]): Average training loss per epoch.
            - val_losses (list[float]): Average validation loss per epoch.
            - val_accs (list[float]): Validation accuracy per epoch.
    """
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        # Training mode
        model.train()
        total_train_loss = 0 #Initialize total_train_loss to 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad() # clear previous gradients
            output = model(inputs) #forward pass
            loss = criterion(output, labels) #compute loss
            loss.backward()  #backward pass
            optimizer.step() #update weights
            total_train_loss += loss.item()

        # Validation mode
        model.eval()
        total_val_loss, total_acc = 0, 0 #Initialize total_val_loss  & total_acc to 0
        with torch.no_grad(): #disable gradient computations
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs) #forward pass
                val_loss = criterion(output, labels) #compute loss
                total_val_loss += val_loss.item()
                preds = torch.argmax(output, dim=1)
                total_acc += (preds == labels).sum().item() / len(labels) # compute accurace

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_acc = total_acc / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_acc)

        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_acc:.4f}')

    return train_losses, val_losses, val_accs


def tune_model_hyperparameters(model_class, model_args, train_data, train_labels, param_grid, epochs, device, k_folds=5):
    """
    Performs hyperparameter tuning for a given model class using grid search and the already implemented training/validation function.

    Args:
        model_class (type): The class of the model to instantiate.
        model_args (dict): Additional fixed arguments to pass when initializing the model.
        train_data : Input data for training.
        train_labels: Labels corresponding to `train_data`.
        param_grid (dict): Dictionary defining the hyperparameter grid with keys corresponding to variable parameters.
        epochs (int): Number of epochs for training each hyperparameter combination.
        device (str): Device to train the model on.

    Returns:
        tuple: 
            - best_params (dict): Dictionary of the best hyperparameter configuration.
            - best_accuracy (float): Highest validation accuracy achieved with the best configuration.
    """
    #Initialize best_accuracy and best_paramas variables
    best_accuracy = 0 
    best_params = {}
    kfold = KFold(n_splits=k_folds)

    # Generate all combinations of hyperparameters from param_grid
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    for params in param_combinations:
        fold_accuracies = []
        # Combine param combination with fixed args
        param_dict = dict(zip(param_names, params))
        model_args_with_params = {**model_args, **param_dict}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data, train_labels)):
            
            print(f"Training fold {fold+1}/{k_folds} with parameters: "
                f"embedding_dim={param_dict.get('embedding_dim')}, "
                f"hidden_dim={param_dict.get('hidden_dim')}, "
                f"n_layers={param_dict.get('n_layers')}, "
                f"learning_rate={param_dict.get('learning_rate')}")
            
            # Split the data into training and validation sets based on the current fold
            X_train_k, X_val_k = X_train[train_idx], X_train[val_idx]
            y_train_k, y_val_k = y_train[train_idx], y_train[val_idx]

            # Create DataLoader for training and validation sets
            train_loader = torch.utils.data.DataLoader(
                            torch.utils.data.TensorDataset(torch.tensor(X_train_k), torch.tensor(y_train_k)),
                            batch_size=32, shuffle=True
                        )
            val_loader = torch.utils.data.DataLoader(
                            torch.utils.data.TensorDataset(torch.tensor(X_val_k), torch.tensor(y_val_k)),
                            batch_size=32, shuffle=False
                        )
        
            # Initialize the model
            model = model_class(**model_args_with_params).to(device)
            optimizer = optim.Adam(model.parameters(), lr=param_dict.get("learning_rate", 0.001))
            criterion = nn.CrossEntropyLoss()

            # Call the train_and_validate function to train the model and get validation results
            train_losses, val_losses, val_accs = train_and_validate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader= train_loader, 
                val_loader = val_loader,
                epochs=epochs,
                device=device
            )

            # Get the best validation accuracy for this set of hyperparameters
        fold_accuracies.append(val_accs[-1])
        
    avg_accuracy = np.mean(fold_accuracies)
        
        # Save best parameters
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_params = param_dict

    return best_params, best_accuracy



# Plotting function
def plot_losses(model_name, train_losses, val_losses, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Losses for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
