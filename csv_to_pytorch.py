import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def make_transform(model_name):
    if model_name == "alexnet":
        return transforms.Compose([
            transforms.Resize((227, 227)),  # Ensure consistent resizing
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure consistent resizing
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def dataset_maker(model_name, batch_size):
    class YoloClassificationDataset(Dataset):
        def __init__(self, csv_file, split, transform=None):
            self.data = pd.read_csv(csv_file)
            self.data = self.data[self.data['split'] == split]  # Filter by split
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            img_path = row['image_path']
            class_label = int(row['class_label'])

            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, class_label

    # Paths
    csv_file = 'dataset/dataset_labels.csv'

    # Transforms
    transform = make_transform(model_name)

    # Make datasets
    train_dataset = YoloClassificationDataset(csv_file, split='train', transform=transform)
    val_dataset = YoloClassificationDataset(csv_file, split='validation', transform=transform)
    test_dataset = YoloClassificationDataset(csv_file, split='test', transform=transform)

    # Make loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    train_loader, val_loader, test_loader = dataset_maker("alexnet",128)
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        print(labels.shape)
        if i == 0:  # Get the first batch
            for j in range(len(images)):  # Loop through all 32 images in the batch
                image = images[j]
                label = labels[j].item()  # Get the label for the current image
                # Convert the tensor image back to a PIL image for visualization
                image_pil = transforms.ToPILImage()(image)  # Convert from tensor to PIL Image

                # Plot the image
                plt.imshow(image_pil)
                plt.title(f"Label: {label}")
                plt.axis('off')  # Turn off axis
                plt.show()

                # Close the previous image plot before showing the next one
                plt.close()

            break  # Break after the first batch


if __name__ == '__main__':
    main()
