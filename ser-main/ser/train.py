import torch.nn.functional as F
import torch
          
def run_training(model, epochs: int, optimizer, test_data, val_data, device):
    best_valid_loss = float('inf')


    for epoch in range(epochs):
        model, loss = _train_batch(model, optimizer, test_data, device)

        print(f"Train Epoch: {epoch} "
                f"| Loss: {loss.item():.4f}")

        # validate
        val_loss, val_acc = _validate_batch(model, val_data, device)
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            print('Saving the model, improvement in validation loss achieved')
            best_model = model

        print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")
    
    return best_model


def _train_batch(model, optimizer, data, device): 
    for i, (images, labels) in enumerate(data):
        images, labels = images.to(device), labels.to(device)        
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    
    return model, loss


def _validate_batch(model, data, device):
    val_loss = 0
    correct = 0
    with torch.no_grad():
      for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            model.eval()
            output = model(images)
            val_loss += F.nll_loss(output, labels, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
      val_loss /= len(data.dataset)
      val_acc = correct / len(data.dataset)
        
    return val_loss, val_acc