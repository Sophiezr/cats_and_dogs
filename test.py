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
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import warnings
warnings.filterwarnings("ignore")
def get_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir', nargs='?', default='/media/sophie/SDD_Data/Data/cats_and_dogs/data/',
                        help='Image directory containing images and masks.')

    parser.add_argument('-cpu_workers', nargs='?', type=int, default=0, help='Number of cpu workers used for '
                                                                             'creating data loaders')

    parser.add_argument('-batch_size', nargs='?', type=int, default=2)
    options = parser.parse_args()
    return options


def softmax(pred):
    """
        Softmax function to convert model prediction values to probabilities.
    """

    return np.array([np.exp(x) for x in pred]) / np.sum([np.exp(x) for x in pred])


def test(model, dataset_loader, seg_criterion, class_criterion):
    model.eval()
    print('Test')
    test_running_loss = 0.0
    test_class_loss = 0.0
    test_seg_loss = 0.0
    test_running_acc = 0.0
    counter = 0
    seg_outputs = []
    class_outputs = []
    target_classes = []
    target_masks = []
    with torch.no_grad():
        for i, (images, masks, labels) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
            counter += images.shape[0]

            labels = torch.tensor(labels, dtype=torch.long).to(device)
            images = torch.tensor(images, dtype=torch.float).to(device)
            masks = torch.tensor(masks, dtype=torch.float).to(device)
            # forward pass
            segmentation_output, classification_output = model(images)

            # Calculate loss for segmentation
            segmentation_loss = seg_criterion(segmentation_output, masks)

            # Calculate loss for classification
            labels = labels.view(-1)
            classification_loss = class_criterion(classification_output, labels)
            test_running_acc += (torch.argmax(classification_output, 1) == labels).sum().cpu().numpy()

            classification_prob = [softmax(tmp) for tmp in classification_output.cpu().numpy()]

            seg_outputs.extend(segmentation_output.cpu().numpy())
            class_outputs.extend(classification_prob)
            target_masks.extend(masks.cpu().numpy())
            target_classes.extend(labels.cpu().numpy())

            # Total loss
            loss = segmentation_loss + classification_loss
            test_class_loss += classification_loss.item()
            test_seg_loss += segmentation_loss.item()
            test_running_loss += loss.item()
            if (i + 1) % 4 == 0:
                print('Test', i + 1, 'iterations, segmentation loss: ', np.round(test_seg_loss / counter, 5),
                      ', classification loss', np.round(test_class_loss / counter, 5))
                print('Test', i, 'acc: ', np.round(test_running_acc / counter, 5))
    # find classes with the highest probability
    class_outputs = np.argmax(class_outputs, 1)
    # loss for the complete epoch
    epoch_loss = test_running_loss / counter
    epoch_acc = test_running_acc / counter
    return epoch_loss, epoch_acc, seg_outputs, class_outputs, target_masks, target_classes


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2.0 * intersection) / (union + 1e-7)
    return dice


def mean_dice_coefficient(y_true_list, y_pred_list):
    dice_scores = []
    for i in range(len(y_true_list)):
        dice_scores.append(dice_coefficient(y_true_list[i], y_pred_list[i]))
    mean_dice = np.mean(dice_scores)
    return mean_dice


def segmentations_to_masks(seg_pred, threshold=0.5):
    y_seg_preds = [(seg > threshold).astype(np.uint8) for seg in seg_pred]
    return y_seg_preds


def write_mask_to_file(mask):
    mask = mask * 255
    mask = mask.reshape(mask.shape[1:])
    mask = Image.fromarray(mask)
    mask.save('outputs/pred_masks/mask.jpg', format='JPEG')


def metrics(y_seg_true, y_seg_pred, y_class_true, y_class_pred, class_names):
    mean_dice = mean_dice_coefficient(y_seg_true, y_seg_pred)
    print('**** Mean Dice Coefficient ****:', np.round(mean_dice, 5))
    conf_matrix = confusion_matrix(y_class_true, y_class_pred)
    df_cm = pd.DataFrame(conf_matrix, class_names, class_names)
    plt.figure(figsize=(25, 25))
    sn.heatmap(df_cm, annot=True)
    # Changed the output file names to prevent over writing to the metrics I created.
    plt.savefig('outputs/metrics/breed_detection_conf_matrix_for_test.png')

    report = classification_report(
        y_class_true,
        y_class_pred,
        target_names=class_names
    )
    # format breed classification report
    report = f'Classification Report:\n{report}'
    print('****** Breed Classification Report ******\n')
    print(report)
    print('For detailed analysis see outputs/metrics directory.')
    with open('outputs/metrics/metrics_for_test.txt', 'w') as file:
        file.write('****** Mean Dice Coefficient ******\n')
        file.write(str(np.round(mean_dice, 5)))
        file.write('\n\n')
        file.write('****** Breed Classification Report ******\n')
        file.write(report)


if __name__ == "__main__":

    # TODO: better way to save the parameters and directories is having a config file. We can define a
    #  config.yaml file with input and output directories and parameters (e.g. hyperparameters, epochs, etc.). This
    #  way, we can easily tweak them in the config file without the need to hardcode them.
    # parameters
    NUM_CLASSES = 37
    img_size = (256, 256)

    # input paths
    # Changed the dataset to single-pet images to prevent errors in case one did not want to split the images
    # and write them to file before testing the model. Please note that this script is to only test model outputs.
    # End-to-end test including two-pet image detection is provided in inference.py.
    test_path = 'inputs/csv/df_test_for_test.csv'

    # output paths
    model_path = 'outputs/model/weight'

    breed_class = {'Abyssinian': 0, 'Bengal': 1, 'Birman': 2, 'Bombay': 3, 'British_Shorthair': 4, 'Egyptian_Mau': 5,
                   'Maine_Coon': 6, 'Persian': 7, 'Ragdoll': 8, 'Russian_Blue': 9, 'Siamese': 10, 'Sphynx': 11,
                   'american_bulldog': 12, 'american_pit_bull_terrier': 13, 'basset_hound': 14, 'beagle': 15,
                   'boxer': 16, 'chihuahua': 17, 'english_cocker_spaniel': 18, 'english_setter': 19,
                   'german_shorthaired': 20, 'great_pyrenees': 21, 'havanese': 22, 'japanese_chin': 23, 'keeshond': 24,
                   'leonberger': 25, 'miniature_pinscher': 26, 'newfoundland': 27, 'pomeranian': 28, 'pug': 29,
                   'saint_bernard': 30, 'samoyed': 31, 'scottish_terrier': 32, 'shiba_inu': 33,
                   'staffordshire_bull_terrier': 34, 'wheaten_terrier': 35, 'yorkshire_terrier': 36}
    args = get_inputs()
    img_dir = args.img_dir
    cpu_workers = args.cpu_workers
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    df_test = pd.read_csv(test_path)

    # build dataloaders
    ds_test = data.get_val_test_dataset(img_dir, df_test, img_size)
    loader_test = data.create_test_loader(ds_test, batch_size=batch_size, num_workers=cpu_workers)

    model = model.UNet(num_classes=NUM_CLASSES)
    model = model.cuda()
    print(model)

    if os.path.exists(model_path):
        print('***** Loading Weights *****')
        best_model_cp = torch.load(model_path)
        model.load_state_dict(best_model_cp)
    else:
        print('***** Can not find weights, quitting... *****')
        quit()

    # Define loss functions
    segmentation_criterion = nn.BCEWithLogitsLoss()
    classification_criterion = nn.CrossEntropyLoss()

    test_epoch_loss, test_epoch_acc, seg_preds, class_preds, target_mask, class_target = test(model, loader_test,
                                                                                              segmentation_criterion,
                                                                                              classification_criterion)
    print(f'Test loss: {test_epoch_loss:.5f}')
    y_seg_masks = segmentations_to_masks(seg_preds, threshold=0.5)
    # write_mask_to_file(y_seg_masks[0])
    metrics(y_seg_masks, target_mask, class_target, class_preds, breed_class)

    print()
