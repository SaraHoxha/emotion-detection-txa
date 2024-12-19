import torch
import re
from torch import optim, nn
from sklearn.model_selection import train_test_split, KFold
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchtools.callbacks import EarlyStopping
from torchtools.exceptions import EarlyStoppingException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'rnn' in name:  # For RNN layers
                if 'weight_ih' in name:  # Input-to-hidden weights
                    torch.nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:  # Hidden-to-hidden weights
                    torch.nn.init.orthogonal_(param)  # Orthogonal initialization for hidden-to-hidden weights
            elif 'fc' in name:  # Fully connected layer weights
                torch.nn.init.xavier_uniform_(param)  # Xavier initialization for fully connected layers
        elif 'bias' in name:  # Initialize biases
            torch.nn.init.zeros_(param)  # Initialize biases to zero

def train_and_validate(model, optimizer, criterion, train_loader, val_loader, epochs, device, patience):
    """
    Trains and validates a PyTorch model.
    """
    train_losses, val_losses, val_accs = [], [], []
    early_stopping = EarlyStopping(monitor="val_loss", mode='min', patience=patience)

    for epoch in range(epochs):
        # Training mode
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(torch.long).to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation mode
        model.eval()
        total_val_loss, total_acc = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(torch.long).to(device)
                output = model(inputs)
                val_loss = criterion(output, labels)
                total_val_loss += val_loss.item()
                preds = torch.argmax(output, dim=1)
                total_acc += (preds == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_acc = total_acc / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_acc)

        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_acc:.4f}')
        
        #Simulate the state dictionary
        state = {
            'meters': {
                'val_loss': type('Metric', (), {'value': avg_val_loss})()
            }
        }
    
        # Call EarlyStopping
        try:
            early_stopping.on_epoch_end(None, state)
        except EarlyStoppingException:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses, val_accs, model


def tune_hyperparams(model_class, model_args, train_data, train_labels, param_grid, epochs, device, patience, k_folds=5, batch_size=128):
    """
    Performs hyperparameter tuning using grid search and K-Fold cross-validation.
    """
    best_accuracy = 0
    best_params = {}
    kfold = KFold(n_splits=k_folds)

    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    for params in param_combinations:
        fold_accuracies = []
        param_dict = dict(zip(param_names, params))
        param_dict_without_lr = {k: v for k, v in param_dict.items() if k != 'learning_rate'}
        model_args_with_params = {**model_args, **param_dict_without_lr}

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
            torch.manual_seed(42)
            np.random.seed(42)
            print(f"Fold {fold+1}/{k_folds}, Params: {param_dict}")

            # Split into fold-specific training and validation sets
            X_train_k, X_val_k = train_data[train_idx], train_data[val_idx]
            y_train_k, y_val_k = train_labels[train_idx], train_labels[val_idx]
            
            y_train_k.to(torch.long)
            y_val_k.to(torch.long)

            train_loader = DataLoader(TensorDataset(X_train_k, y_train_k), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_k, y_val_k), batch_size=batch_size, shuffle=False)

            # Initialize model
            model = model_class(**model_args_with_params).to(device)
            initialize_weights(model)
            optimizer = optim.Adam(model.parameters(), lr=param_dict.get("learning_rate", 0.001), weight_decay=param_dict.get("weight_decay",1e-5))
            criterion = nn.CrossEntropyLoss()

            # Train and validate
            _, _, val_accs, _ = train_and_validate(model, optimizer, criterion, train_loader, val_loader, epochs, device, patience)
            fold_accuracies.append(val_accs[-1])

        avg_accuracy = np.mean(fold_accuracies)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = param_dict

    return best_params, best_accuracy


def plot_losses(model_name, dataset_name, train_losses, val_losses, epochs):
    """
    Plots training and validation losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Losses for {model_name} {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train_val_split(train_data, train_labels, val_split=0.2, batch_size=128):
    """
    Splits training data into training and validation sets.
    """
    # Convert train_data and train_labels to numpy arrays if they are not already
    train_data = train_data.numpy() if isinstance(train_data, torch.Tensor) else train_data
    train_labels = train_labels.numpy() if isinstance(train_labels, torch.Tensor) else train_labels

    # Use train_test_split to split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=val_split, random_state=42
    )

    # Convert the numpy arrays back to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    # Create DataLoader for training and validation sets
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def test(model, test_loader, device):
    """
    Tests a PyTorch model.
    """
    predictions, true_labels = [], []
    
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():  # Disable gradient computation for testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels

def compute_metrics(predictions, true_labels): 
    """
    Computes metrics for model performance.
    """
    metrics={}
    
    metrics['accuracy'] = accuracy_score(true_labels, predictions)
    metrics['precision'] = precision_score(true_labels, predictions, average='macro')
    metrics['recall'] = recall_score(true_labels, predictions, average='macro')
    metrics['f1'] = f1_score(true_labels, predictions, average='macro')
    
    metrics['precision_weighted'] = precision_score(true_labels, predictions, average='weighted')
    metrics['recall_weighted'] = recall_score(true_labels, predictions, average='weighted')
    metrics['f1_weighted'] = f1_score(true_labels, predictions, average='weighted')
    
    return metrics

def plot_confusion_matrix(true_labels, predictions, label_classes):
    """
    Plots a confusion matrix.
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_classes, yticklabels=label_classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()