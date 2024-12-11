import torch
import torch.nn as nn
from csv_to_pytorch import dataset_maker
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torchvision.models as models


def train_model(net, train_loader, optimizer, criterion, num_epochs, device, wandb_log_status):
    """
    This is for training the model
    """
    net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Loss computation
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # printing all stats for better monitoring
            running_loss += loss.item()
            print(f"epoch : {epoch + 1} batch : {i + 1}")
            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

            # wandb logging code
            if wandb_log_status:
                wandb.log({"Train Loss": loss, "Train running Loss": running_loss})


def log_wandb_metrics(y_true, y_pred, y_scores, class_names):
    """
    This is for logging every metric and statistics to wandb
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=15)
    wandb.log({"Confusion Matrix": wandb.Image(plt)})
    plt.close()

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Calculate mAP (mean Average Precision) for multi-class classification
    y_true_one_hot = np.eye(len(class_names))[y_true]  # Convert to one-hot encoded format for mAP calculation
    map_score = average_precision_score(y_true_one_hot, y_scores, average='macro')

    # Calculate accuracy
    accuracy = 100 * sum(np.array(y_pred) == np.array(y_true)) / len(y_true)

    # Log precision, recall, F1 score, mAP, and accuracy to wandb
    wandb.summary["Precision"] = precision
    wandb.summary["Recall"] = recall
    wandb.summary["F1 Score"] = f1
    wandb.summary["mAP Score"] = map_score
    wandb.summary["Accuracy"] = accuracy

    # Print the metrics
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision (Weighted): {precision:.4f}')
    print(f'Recall (Weighted): {recall:.4f}')
    print(f'F1 Score (Weighted): {f1:.4f}')
    print(f'mAP (mean Average Precision): {map_score:.4f}')


def test_model_with_metrics(net, test_loader, device, num_classes, wandb_log_status):
    """
    This is for testing the model
    """
    net.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []
    y_scores = []

    correct = 0
    total = 0

    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            # Get predicted classes and probabilities
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect true labels, predicted labels, and prediction scores for metrics calculation
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())

    # Class names (update accordingly for your dataset)
    class_names = [f"Class {i}" for i in range(num_classes)]

    # wandb logging code
    if wandb_log_status:
        log_wandb_metrics(y_true, y_pred, y_scores, class_names)


def main():
    # hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001
    num_classes = 4
    wandb_log_status = True

    # Dataset
    train_loader, val_loader, test_loader = dataset_maker("alexnet", batch_size)

    # Model, Loss function, and Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = models.alexnet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # wandb config and login setup
    config = dict(
        epochs=num_epochs,
        classes=num_classes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dataset="Custom",
        architecture="CNN",
    )
    wandb.login(key="96d766351c53a310e22505aa9e083a4a288df77a")

    # wandb logging
    if wandb_log_status:
        wandb.init(
            # Set the project name
            project="DRDO 3",
            # Set the run name
            name=f"AlexNet",
            # Track hyperparameters and run metadata
            config=config
        )
        wandb.watch(models=net, log_freq=100, log="all")

    # Train the model
    train_model(net, train_loader, optimizer, criterion, num_epochs, device, wandb_log_status)

    # Test the model
    test_model_with_metrics(net, test_loader, device, num_classes, wandb_log_status)


if __name__ == '__main__':
    main()
