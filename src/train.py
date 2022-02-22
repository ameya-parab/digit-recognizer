import os
import typing

import torch
import torch.nn as nn

from .model import NeuralNetwork
from .utils import set_random_seed

MODEL_PATH = os.path.join("..", "model", "cnn.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_training(
    train_dataloader, valid_dataloader, epochs, lr, random_seed, verbose=False
) -> float:

    set_random_seed(random_seed=random_seed)

    model = NeuralNetwork().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):

        model.train()

        for step, (images, labels) in enumerate(train_dataloader):

            images = torch.unsqueeze(images, dim=1).to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step % 1000 == 0) and verbose:

                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                )

        # Save Model
        torch.save(model.state_dict(), MODEL_PATH)

        # Evaluate Model
        accuracy, _ = evaluate(model=model, dataloader=valid_dataloader)

        if (epoch == 0) or (((epoch+1) % 1) == 0) or (epoch+1==epochs):

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}"
            )

    return accuracy


def evaluate(model, dataloader) -> typing.Union[float, torch.Tensor]:

    model.eval()
    with torch.no_grad():

        batch_predictions, batch_ground_labels = [], []
        for _, (batch_images, batch_labels) in enumerate(dataloader):

            batch_images = torch.unsqueeze(batch_images, dim=1).to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            outputs = model(batch_images)
            predicted = torch.argmax(outputs, dim=1)

            batch_predictions.append(predicted)
            batch_ground_labels.append(batch_labels)

        predictions = torch.cat(batch_predictions, dim=0)
        labels = torch.cat(batch_ground_labels, dim=0)

        correct_predictions = torch.eq(predictions, labels).sum().item()
        total_images = predictions.shape[0]

        return ((correct_predictions / total_images) * 100, predictions)
