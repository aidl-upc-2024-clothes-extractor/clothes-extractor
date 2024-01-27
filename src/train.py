import torch
import torch.optim as optim
from torch.nn import L1Loss

def train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate):
    criterion = L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, inputs in enumerate(train_dataloader.next_batch()):
            target = inputs["predict"]
            source = inputs["img_masked"]
            optimizer.zero_grad()

            outputs = model(source)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(val_dataloader.next_batch()):
                target = inputs["predict"]

                outputs = model(inputs)
                loss = criterion(outputs, target)

                running_loss += loss.item()

        avg_val_loss = running_loss / len(val_dataloader)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}')

    print('Finished Training')
    return model
