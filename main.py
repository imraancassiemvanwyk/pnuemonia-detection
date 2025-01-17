import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


data_dir = '/kaggle/input/nih-chest-x-rays/NIH chest x-rays'
images_dir = os.path.join(data_dir, 'images')
csv_path = os.path.join(data_dir, 'Data_Entry_2017.csv')


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_label="Pneumonia"):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_label = target_label
        # Create binary labels for target
        self.data['label'] = self.data['Finding Labels'].apply(
            lambda x: 1 if target_label in x else 0
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['Image Index'])
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        label = self.data.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = ChestXrayDataset(csv_file=csv_path, img_dir=images_dir, transform=transform_train)
val_dataset = ChestXrayDataset(csv_file=csv_path, img_dir=images_dir, transform=transform_val)


train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)  # Adjust for binary classification
model = model.to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

  
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

     
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

       
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend((preds > 0.5).astype(int))
        all_labels.extend(labels.cpu().numpy())

    train_accuracy = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds)
    train_recall = recall_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds)
    train_auc = roc_auc_score(all_labels, [p[0] for p in all_preds])

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
          f"Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, "
          f"Recall: {train_recall:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")

    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend((preds > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds)
    val_recall = recall_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds)
    val_auc = roc_auc_score(all_labels, [p[0] for p in all_preds])

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}, "
          f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
