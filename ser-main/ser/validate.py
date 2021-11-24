import torch
import torch.nn.functional as F


def validate_batch(model, data, device):
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
        
    return [val_loss, val_acc]