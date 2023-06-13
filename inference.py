import torch
import cv2
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
from torchvision import ops, transforms as trans
import torchvision
from torchvision.transforms import functional as F
import warnings

warnings.filterwarnings("ignore")


def get_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir', nargs='?', default='/media/sophie/SDD_Data/Data/cats_and_dogs/data/',
                        help='Image directory containing images and masks.')

    options = parser.parse_args()
    return options


def softmax(pred):
    """
        Softmax function to convert model prediction values to probabilities.
    """

    return np.array([np.exp(x) for x in pred]) / np.sum([np.exp(x) for x in pred])


def segmentations_to_masks(seg_pred, threshold=0.5):
    y_seg_preds = [(seg > threshold).astype(np.uint8) for seg in seg_pred]
    return y_seg_preds


def write_mask_to_file(mask, name):
    mask = mask * 255
    mask = mask.reshape(mask.shape[1:])
    mask = np.rollaxis(mask, 1, 0)
    mask = Image.fromarray(mask)
    mask.save('outputs/pred_masks/' + name + '.jpg', format='JPEG')

def merge_masks(mask_0, mask_1):
    # merged_mask = np.concatenate((mask_0, mask_1), axis =1)
    merged_mask = cv2.hconcat([mask_0, mask_1])
    return merged_mask
def preprocess(img, size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = trans.Compose([

        trans.Resize(size),
        trans.ToTensor(),
        trans.Normalize(mean=mean, std=std)
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    img = torch.tensor(img, dtype=torch.float).to(device)

    return img


def calculate_area(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    length = y2 - y1
    return width * length


def find_two_biggest_bbox(bbox):
    # Calculate areas of bounding boxes
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bbox]

    # Sort the bounding boxes based on area (descending order)
    sorted_boxes = sorted(zip(bbox, areas), key=lambda x: x[1], reverse=True)

    # Get the two biggest bounding boxes
    big_box_0 = sorted_boxes[0][0]
    big_box_1 = sorted_boxes[1][0]
    bboxes = [big_box_0, big_box_1]
    return bboxes


def calculate_iou(box1, box2):
    # Extract the coordinates of the boxes
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1_box1, x1_box2)
    y_top = max(y1_box1, y1_box2)
    x_right = min(x2_box1, x2_box2)
    y_bottom = min(y2_box1, y2_box2)

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Calculate the area of both bounding boxes
    box1_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    box2_area = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

    # Calculate the union area by subtracting the intersection area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area
    return iou


def get_num_of_pets(bbox):
    """
    Set of rules to detect number of pets in an image.
    """

    if len(bbox) > 2:
        # find two biggest bounding boxes if more than two objects are detected.
        bbox = find_two_biggest_bbox(bbox)
    if len(bbox) == 0:
        # we know there is at least one pet in the image even if it is not detected
        num_of_pets = 1
    elif len(bbox) == 1:
        num_of_pets = 1
    else:
        area_0 = calculate_area(bbox[0])
        area_1 = calculate_area(bbox[1])
        IoU = calculate_iou(bbox[0], bbox[1])
        if IoU > 0.5:
            num_of_pets = 1
        # if one bbox is much smaller than the other, it is a random bbox.
        elif area_1 < area_0 / 2 or area_0 < area_1 / 2:
            num_of_pets = 1
        else:
            num_of_pets = 2

    return num_of_pets


def detect_num_of_pets(model, img):
    image_tensor = F.to_tensor(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    with torch.no_grad():
        predictions = model(image_tensor)
    # Process the predictions
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']

    # Filter predictions
    cat_dog_indices = [i for i, score in enumerate(scores) if score > 0.85]
    filtered_boxes = boxes[cat_dog_indices]
    pets = get_num_of_pets(filtered_boxes.numpy())
    if debug:
        numpy_image = np.array(img)
        cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        for i, box in enumerate(filtered_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if len(cat_dog_indices):
            # cv2.imwrite('output/test_rcnn/' + str(i) + '.jpg', cv_image)
            cv2.imwrite('/ media/sophie/SDD_Data/Data/cats_and_dogs/frcnn_test/' + str(i) + '.jpg', cv_image)

        else:
            print('no box found')
    return pets


def split_image(img):

    width, height = img.size
    # Split the image in half vertically
    left_img = img.crop((0, 0, width // 2, height))
    right_img = img.crop((width // 2, 0, width, height))

    return [left_img, right_img]


def inference(model, img):
    img = preprocess(img, img_size)
    model.eval()
    with torch.no_grad():
        segmentation_output, classification_output = model(img)
        y_seg_masks = segmentations_to_masks(segmentation_output.cpu().numpy(), threshold=0.5)
        classification_prob = [softmax(tmp) for tmp in classification_output.cpu().numpy()]
        class_output = np.argmax(classification_prob, 1)[0]

    return y_seg_masks[0], class_output


if __name__ == "__main__":

    # TODO: better way to save the parameters and directories is having a config file. We can define a
    #  config.yaml file with input and output directories and parameters (e.g. hyperparameters, epochs, etc.). This
    #  way, we can easily tweak them in the config file without the need to hardcode them.
    # parameters
    NUM_CLASSES = 37
    img_size = (256, 256)
    debug = False
    num_masks_to_file = 8
    # input paths
    test_path = 'inputs/csv/df_test.csv'

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
    breeds = list(breed_class.keys())
    args = get_inputs()
    img_dir = args.img_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    df_test = pd.read_csv(test_path)

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
    # load torchvision object detection model to detect number of pets in the image.
    od_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    od_model.eval()
    for i, row in df_test.iterrows():
        seg_masks = []

        image = Image.open(img_dir + row['id'] + '/image.jpg')
        n_pets = detect_num_of_pets(od_model, image)
        if n_pets > 1:
            images = split_image(image)
        else:
            images = [image]

        print('Image ID: ', row['id'])
        if n_pets > 1:
            print('Image of a cat and dog')

        for img in images:
            seg_mask, class_out = inference(model, img)
            seg_masks.append(seg_mask)

            # ****** outputs ******
            if class_out <= 11 and n_pets == 1:
                print('Image of a cat.')
            elif class_out > 11 and n_pets == 1:
                print('Image of a dog.')

            print('Breed: ', breeds[class_out])
        if i <= num_masks_to_file:
            # Only write limited number of images and masks
            if n_pets > 1:
                mask = merge_masks(seg_masks[0], seg_masks[1])
                write_mask_to_file(mask, row['id'])
            else:
                write_mask_to_file(seg_mask, row['id'])

        print(
                'Only limited number of masks are writen to file. To increase the limit, change num_masks_to_file in inference.py')
        print('******************************************************************************************************************')
    print('***** INFERENCE COMPLETE *****')
