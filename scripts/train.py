import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import create_model
from dataset import CarPartDataset

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = create_model(num_classes=NUM_CLASSES)
    model.to(device)

    dataset = CarPartDataset("dataset/train/images", "dataset/train/annotations.json", transforms=ToTensor())
    dataset_val = CarPartDataset("dataset/val/images", "dataset/val/annotations.json", transforms=ToTensor())

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {losses.item():.4f}")
