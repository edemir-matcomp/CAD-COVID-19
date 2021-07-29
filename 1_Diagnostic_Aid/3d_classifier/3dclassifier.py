"""
3D Convolutional Neural Network for CT scans classification (COVID and Non-COVID)
"""

# Imports and setup
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from warnings import simplefilter

from monai.data import CacheDataset, DataLoader, ImageDataset
from monai.transforms import (
    AddChannel,
    Compose,
    ToTensor,
    RandFlip
)

from models import resnet3dd

# version name
version = "version_name"

"""
Data directory must have two folders named abnormal (covid positive) and
normal (covid negative) each one of them with sub-folders containing
multiples datasets
"""
# diretorio dados

#data_dir = "/mnt/DADOS_GRENOBLE_1/dataset_covid"
data_dir = "/mnt/CADCOVID/dataset_covid_global"

"""
Read image filenames from the dataset folders
Reading the paths of the CT scans from directories.
"""

abnormal_files, normal_files = [], []
abnormal_files = [os.path.join(data_dir, "abnormal", source, x)
                for source in os.listdir(os.path.join(data_dir, "abnormal"))
                for x in os.listdir(os.path.join(data_dir, "abnormal", source))]
normal_files = [os.path.join(data_dir, "normal", source, x)
                for source in os.listdir(os.path.join(data_dir, "normal"))
                for x in os.listdir(os.path.join(data_dir, "normal", source))]

num_class = 2
abnormal_labels = np.array([0 for _ in range(len(abnormal_files))])
normal_labels = np.array([1 for _ in range(len(normal_files))])

print("all abnormal files: ", len(abnormal_files))
print("all normal files: ", len(normal_files))
print("abnormal_labels: ", len(abnormal_labels))
print("normal_labels: ", len(normal_labels))

file_list, label_list = [], []
file_list.extend(abnormal_files)
file_list.extend(normal_files)
label_list.extend(abnormal_labels)
label_list.extend(normal_labels)

print("total labels: ", len(label_list))
print("total files: ", len(file_list))

#Prepare training, validation and test data lists
trainX, trainY = [], []
valX, valY = [], []
testX, testY = [], []

# dataset division for training, validation and test
valid_frac, test_frac = 0.2, 0.0
for i in range(len(label_list)):
    rann = np.random.random()
    if rann < valid_frac:
        valX.append(file_list[i])
        valY.append(label_list[i])
    elif rann < test_frac + valid_frac:
        testX.append(file_list[i])
        testY.append(label_list[i])
    else:
        trainX.append(file_list[i])
        trainY.append(label_list[i])

print("Training count =", len(trainX), "Validation count =", len(valX), "Test count =",len(testX))
print("Training abnormal samples: ", trainY.count(0), " - Training normal samples: ", trainY.count(1))
print("Validation abnormal samples: ", valY.count(0), "- Validation normal samples: ", valY.count(1))

"""
Define transforms, Dataset and Dataloader to preprocess data

CT scans store raw voxel intensity in Hounsfield units (HU).
They range from lower than -1024 to above 2000. Above 400 are bones
with different radiointensity, so this is used as a higher bound.
A threshold between -1000 and 400 is commonly used to normalize CT scans.

Rotate the volumes by 90 degrees, so the orientation is fixed
Scale the HU values to be between 0 and 1.
Resize width, height and depth.
"""

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 50
    desired_width = 450
    desired_height = 450
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

transforms_train = Compose([
    AddChannel(),
    RandFlip(prob=0.5, spatial_axis=0),
    ToTensor()
])

transforms_val = Compose([
    AddChannel(),
    ToTensor()
])

class MMDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        sample = process_scan(self.image_files[idx])
        sample = self.transforms(sample)
        return sample, self.labels[idx]

"""
Create training, validation and test dataloaders
"""

train_ds = MMDataset(trainX, trainY, transforms_train)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4,
                            pin_memory=torch.cuda.is_available())

val_ds = MMDataset(valX, valY, transforms_val)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4,
                            pin_memory=torch.cuda.is_available())

test_ds = MMDataset(testX, testY, transforms_val)
test_loader = DataLoader(test_ds, batch_size=2, num_workers=4,
                            pin_memory=torch.cuda.is_available())

"""
Define network and optimizer
"""

# Create cnn model
torch.cuda.set_device("cuda:2")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
model = resnet3dd.generate_model(
    model_depth=18,
    n_input_channels=1,
    n_classes=num_class
    ).to(device)

# Define loss function, optimizer and lr scheduler
weights = torch.tensor([0.17, 0.83]).cuda()
loss_function = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
                                weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)

# start a typical PyTorch training
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter(comment=version)
# training epochs
max_epochs = 100

epoch_vloss_values = []
metric_bacc_values = []
metric_bacc_train_values = []
y_true_values = []
y_pred_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")

    # print learning rate
    for param_group in optimizer.param_groups:
        print("lr: ", param_group['lr'])

    model.train()
    epoch_loss = 0
    step = 0

    y_true_train = []
    y_pred_train = []

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Perform forward pass
        outputs = model(inputs)
        # Compute loss
        loss = loss_function(outputs, labels)
        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        if (step % 100) == 0:
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        y_true_train.extend(labels.tolist())
        y_pred_train.extend((outputs.argmax(dim=1)).tolist())

    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    # compute balanced accuracy
    weights_by_class = [4.88 if y==1 else 1 for y in y_true_train]
    metric_bacc_train = balanced_accuracy_score(y_true_train, y_pred_train, weights_by_class, adjusted=False)
    metric_bacc_train_values.append(metric_bacc_train)
    writer.add_scalar("train/balanced_accuracy", metric_bacc_train, epoch + 1)
    print(f"epoch {epoch + 1} train accuracy: {metric_bacc_train:.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    writer.add_scalar("train/loss", epoch_loss, epoch + 1)
    print(f"epoch {epoch + 1} train average loss: {epoch_loss:.4f}")
    lr_scheduler.step()

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            num_correct = 0.0
            epoch_vloss = 0
            val_step = 0
            y_true = []
            y_pred = []

            for val_data in val_loader:
                val_step += 1
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_images)

                vloss = loss_function(val_outputs, val_labels)
                epoch_vloss += vloss.item()
                epoch_val_len = len(val_ds) // val_loader.batch_size
                if (step % 50) == 0:
                    print(f"{val_step}/{epoch_val_len}, val_loss: {vloss.item():.4f}")

                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                num_correct += value.sum().item()

                y_true.extend(val_labels.tolist())
                y_pred.extend((val_outputs.argmax(dim=1)).tolist())

            epoch_vloss /= val_step
            epoch_vloss_values.append(epoch_vloss)
            writer.add_scalar("validation/loss", epoch_vloss, epoch + 1)
            print(f"epoch {epoch + 1} val average loss: {epoch_vloss:.4f}")

            # ignore all future warnings
            simplefilter(action='ignore', category=FutureWarning)
            # compute balanced accuracy
            weights_by_class = [4.88 if y==1 else 1 for y in y_true]
            metric_bacc = balanced_accuracy_score(y_true, y_pred, weights_by_class, adjusted=False)
            metric_bacc_values.append(metric_bacc)
            writer.add_scalar("validation/balanced_accuracy", metric_bacc, epoch + 1)
            print(f"epoch {epoch + 1} val accuracy: {metric_bacc:.4f}")

            if metric_bacc > best_metric and epoch > 50:
                best_metric = metric_bacc
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(),
                           'best_model'+version+'.pth')
                print("saved new best metric model")
            print(
                "current epoch: {} current accuracy: {:.4f} "
                "best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, metric_bacc, best_metric, best_metric_epoch
                )
            )

            # compute confusion matrix
            if (epoch + 1) == max_epochs:
                y_true_values.extend(y_true)
                y_pred_values.extend(y_pred)
                cm = confusion_matrix(y_true_values, y_pred_values)
                disp = ConfusionMatrixDisplay(cm, display_labels=("0 - abnormal","1 - normal"))
                disp.plot()
                plt.savefig('cm'+version+'.png')

print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")
writer.close()
