import torch

def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    # Determine if a GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Move the model to the appropriate device
    model.to(device)
    
    # Set the model to training mode
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        for (batch, labels) in train_loader:
            # Move data to the active device (GPU/CPU)
            batch = batch.to(device)
            labels = labels.to(device)
            
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(batch)
            
            # 3. Calculate the loss
            loss = loss_fn(outputs, labels)
            
            # 4. Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # 5. Perform a single optimization step (parameter update)
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += batch.size(0)
            
        # Calculate epoch-level metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions.double() / total_samples
        
        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")
        
    print("Training complete!")
    return model
