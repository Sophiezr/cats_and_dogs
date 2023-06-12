import torch
import torch.nn as nn
import torch.optim as optim
import model
import data
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")
class SaveBestModel:
    """
        Class to save the best model while training. If the current epoch's
        validation loss is less than the previous least loss, then save the
        model state.
    """

    def __init__(
            self, best_valid_loss=float('inf'),
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss, epoch, model, optimizer, output
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save(model.state_dict(), output)


def get_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir', nargs='?', default='/media/sophie/SDD_Data/Data/cats_and_dogs/data/',
                        help='Image directory containing images and masks.')

    parser.add_argument('-cpu_workers', nargs='?', type=int, default=20, help='Number of cpu workers used for '
                                                                                      'creating data loaders')

    parser.add_argument('-batch_size', nargs='?', type=int, default=32)
    options = parser.parse_args()
    return options

def save_acc_plot(train_acc, valid_acc, plot_path):
    """
        Function to save accuracy plot to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(plot_path, transparent=False)


def save_loss_plot(train_loss, valid_loss, plot_path):
    """
        Function to save accuracy plot to disk.
    """

    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_path, transparent=False)


def train(model, dataset_loader, optimizer, seg_criterion, class_criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_class_loss = 0.0
    train_seg_loss = 0.0
    train_running_acc = 0.0
    counter = 0
    for i, (images, masks, labels) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
        counter += images.shape[0]

        labels = torch.tensor(labels, dtype=torch.long).to(device)
        images = torch.tensor(images, dtype=torch.float).to(device)
        masks = torch.tensor(masks, dtype=torch.float).to(device)

        optimizer.zero_grad()

        # Forward pass
        segmentation_output,  classification_output = model(images)

        # Calculate loss for segmentation
        segmentation_loss = seg_criterion(segmentation_output, masks)

        # Calculate loss for classification
        labels = labels.view(-1)
        classification_loss = class_criterion(classification_output, labels)
        train_running_acc += (torch.argmax(classification_output, 1) == labels).sum().cpu().numpy()
        # Total loss
        loss = segmentation_loss + classification_loss
        train_class_loss += classification_loss.item()
        train_seg_loss += segmentation_loss.item()
        train_running_loss += loss.item()

        # Backpropagation
        loss.backward()

        # Update the optimizer parameters
        optimizer.step()
        if (i + 1) % 32 == 0:
            print('train', i + 1, 'iterations, segmentation loss: ', np.round(train_seg_loss / counter, 5), ', classification loss', np.round(train_class_loss / (counter), 5))
            print('train', i, 'acc: ', np.round(train_running_acc / counter, 5))
    # Loss for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = train_running_acc / counter
    return epoch_loss, epoch_acc


def validate(model, dataset_loader, seg_criterion, class_criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_class_loss   = 0.0
    valid_seg_loss     = 0.0
    valid_running_acc  = 0.0
    counter = 0
    with torch.no_grad():
        for i, (images, masks, labels) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
            counter += images.shape[0]

            labels = torch.tensor(labels, dtype=torch.long).to(device)
            images = torch.tensor(images, dtype=torch.float).to(device)
            masks = torch.tensor(masks, dtype=torch.float).to(device)
            # forward pass
            segmentation_output,  classification_output = model(images)

            # Calculate loss for segmentation
            segmentation_loss = seg_criterion(segmentation_output, masks)

            # Calculate loss for classification
            labels = labels.view(-1)
            classification_loss = class_criterion(classification_output, labels)
            valid_running_acc += (torch.argmax(classification_output, 1) == labels).sum().cpu().numpy()
            # Total loss
            loss = segmentation_loss + classification_loss
            valid_class_loss   += classification_loss.item()
            valid_seg_loss     += segmentation_loss.item()
            valid_running_loss += loss.item()
            if (i + 1) % 8 == 0:
                print('validate', i+1 , 'iterations, segmentation loss: ', np.round(valid_seg_loss/counter, 5), ', classification loss', np.round(valid_class_loss/counter, 5))
                print('Validate', i, 'acc: ', np.round(valid_running_acc / counter, 5))
    # loss for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = valid_running_acc / counter
    return epoch_loss, epoch_acc

if __name__ == "__main__":

    # TODO: better way to save the parameters and directories is having a config file. We can define a
    #  config.yaml file with input and output directories and parameters (e.g. hyperparameters, epochs, etc.). This
    #  way, we can easily tweak them in the config file without the need to hardcode them.
    # parameters
    NUM_CLASSES = 37
    learning_rate = 0.001
    epochs = [50, 30]
    img_size = (256, 256)

    # input paths
    # Changed the dataset to single-pet images to prevent errors in case one did not want to split the images
    # and write them to file before testing the model train.
    train_path = 'inputs/csv/df_train_for_test.csv'
    test_path  = 'inputs/csv/df_val_for_test.csv'

    # output paths
    # Changed the name of weight and plots to prevent writing to the weight and plots I used for calculating metrics.
    model_path = 'outputs/model/weight_for_test'
    loss_plot_path = 'outputs/plot/loss_for_test.png'
    acc_plot_path = 'outputs/plot/acc_for_test.png'

    args = get_inputs()
    img_dir = args.img_dir
    cpu_workers = args.cpu_workers
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(test_path)

    # build dataloaders
    ds_train = data.get_train_dataset(img_dir, df_train, img_size)
    ds_val = data.get_train_dataset(img_dir, df_val, img_size)
    loader_train, loader_val = data.create_data_loaders(ds_train, ds_val, batch_size=batch_size, num_workers=cpu_workers)

    model = model.UNet(num_classes=NUM_CLASSES)
    model = model.cuda()
    print(model)

    if os.path.exists(model_path):
        print('***** Loading Weights *****')
        best_model_cp = torch.load(model_path)
        model.load_state_dict(best_model_cp)

    # Define loss functions
    segmentation_criterion = nn.BCEWithLogitsLoss()
    classification_criterion = nn.CrossEntropyLoss()

    save_best_model = SaveBestModel()

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    for rank in range(len(epochs)):

        learning_rate = learning_rate / (10 ** rank)
        print('learning rate:', learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs[rank]):
            print(f"[INFO]: Epoch {epoch + 1} of {epochs[rank]}")

            train_epoch_loss, train_epoch_acc = train(model, loader_train, optimizer, segmentation_criterion, classification_criterion)
            valid_epoch_loss, valid_epoch_acc = validate(model, loader_val, segmentation_criterion, classification_criterion)
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)

            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)

            print(f"Training loss: {train_epoch_loss:.5f}")
            print(f"Validation loss: {valid_epoch_loss:.5f}")
            # save the best weight till now if we have the least loss in the current epoch
            save_best_model(valid_epoch_loss, epoch, model, optimizer,model_path)

            save_loss_plot(train_loss, valid_loss, loss_plot_path)
            save_acc_plot(train_acc, valid_acc, acc_plot_path)
            print('-' * 50)

    print('EPOCH COMPLETE')

