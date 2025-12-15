import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import timm
import os
from safetensors.torch import load_file
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
DATA_DIR = 'Data'
MODEL_SAVE_PATH = 'mobilevit_emotion_recognition.pth'
PLOT_SAVE_PATH = 'training_plot.png'


# --- Plotting Function ---
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot training & validation accuracy values
    ax1.plot(history['train_acc'])
    ax1.plot(history['val_acc'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss values
    ax2.plot(history['train_loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Val'], loc='upper left')

    plt.savefig(PLOT_SAVE_PATH)
    plt.close()
    print(f"Training plot saved to {PLOT_SAVE_PATH}")


# --- 5. Test Function ---
def test_model(model, test_loader, device, criterion):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    # Disable gradient calculations
    with torch.no_grad():
        # Iterate over data with a progress bar
        progress_bar = tqdm(test_loader, desc='Testing')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Use autocast for mixed precision inference
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix(loss=running_loss / ((progress_bar.n + 1) * BATCH_SIZE), acc=running_corrects.double() / ((progress_bar.n + 1) * BATCH_SIZE))

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    print('\n--- Test Results ---')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    print('--------------------')


# --- 6. Training and Validation Loop ---

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # Initialize GradScaler for mixed precision
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    best_model_wts = model.state_dict()
    best_acc = 0.0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data with a progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}/{num_epochs}')
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for a small performance gain

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Use autocast for mixed precision
                    with autocast(enabled=torch.cuda.is_available()):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer) # unscale gradients before clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                progress_bar.set_postfix(loss=running_loss / ((progress_bar.n + 1) * BATCH_SIZE), acc=running_corrects.double() / ((progress_bar.n + 1) * BATCH_SIZE))


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # Save the best model
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"Best model saved to {MODEL_SAVE_PATH} with accuracy: {best_acc:.4f}")
        
        scheduler.step()

    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights and plot history
    model.load_state_dict(best_model_wts)
    plot_training_history(history)
    return model

if __name__ == '__main__':
    # Check for GPU availability and define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable cuDNN benchmarking for performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # --- 1. Data Loading and Preprocessing ---

    # Define transformations for the images
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the dataset from the 'Data' directory
    # We use the 'train' transform initially for all images, as it includes ToTensor.
    # The transform for val/test sets will be corrected after splitting.
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # --- 2. Dataset Splitting (Train, Validation, Test) ---
    
    # Define split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    # test_ratio is the remainder
    
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Splitting dataset into: \n- Training: {train_size} images\n- Validation: {val_size} images\n- Testing: {test_size} images")

    # Split the dataset
    # Use a fixed generator for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # for reproducibility
    )

    # Apply the correct (non-augmenting) transformations to the validation and test sets
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['val']

    # --- 3. Create DataLoaders ---
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")


    # --- 4. Model Definition ---

    # Load a pretrained MobileViT model (e.g., 'mobilevit_s')
    # We will replace the final classifier layer to match our number of classes
    model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)

    # --- Manually Load Pretrained Weights ---
    LOCAL_WEIGHTS_PATH = 'model.safetensors'
    if os.path.exists(LOCAL_WEIGHTS_PATH):
        print(f"Loading pretrained weights from local file: {LOCAL_WEIGHTS_PATH}")
        state_dict = load_file(LOCAL_WEIGHTS_PATH, device="cpu")
        
        # Remove the final classification layer's weights from the pretrained model
        # as its size (1000 classes) does not match our model's size (5 classes).
        if 'head.fc.weight' in state_dict:
            del state_dict['head.fc.weight']
        if 'head.fc.bias' in state_dict:
            del state_dict['head.fc.bias']

        # Load the modified state dict. `strict=False` is good practice for transfer learning.
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    else:
        print(f"WARNING: Pretrained weights file not found at '{LOCAL_WEIGHTS_PATH}'.")
        print("The model will be trained from scratch.")
        print(f"Please ensure 'model.safetensors' is in the same directory as train.py.")

    model.to(device) # Move model to device after loading weights

    # --- 5. Training Setup ---

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- 6. Start Training & Evaluation ---
    try:
        trained_model = train_model(model, criterion, optimizer, scheduler, num_epochs=EPOCHS)
        
        print("\n--- Starting Final Evaluation on the Test Set ---")
        # The train_model function already loads the best model weights into the model object
        test_model(trained_model, test_dataloader, device, criterion)

    except Exception as e:
        print(f"\nAn error occurred during training or evaluation: {e}")
        print("Please ensure you have a 'Data' directory with subdirectories for each emotion class.")
        print("Also, consider reducing BATCH_SIZE if you are running out of memory.")