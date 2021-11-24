import torch.nn.functional as F
          
def train_batch(model, optimizer, data, device): 
    for i, (images, labels) in enumerate(data):
        images, labels = images.to(device), labels.to(device)        
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    
    return model, loss