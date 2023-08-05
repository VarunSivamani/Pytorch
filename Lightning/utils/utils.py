import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def show_transform(input_tensor):
  inv_normalize = transforms.Normalize(
      mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
      std=[1/0.23, 1/0.23, 1/0.23]
  )

  img = input_tensor.squeeze(0).to('cpu')
  img = inv_normalize(img)
  rgb_img = np.transpose(img, (1, 2, 0))
  rgb_img = rgb_img.numpy()

  return rgb_img

def plot_examples(images, labels, figsize=(20,10)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle("GradCam Images with Labels", fontsize=16)

    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.tight_layout()
        image = images[i]
        plt.imshow(image, cmap='gray')
        label = labels[i]
        plt.title(str(label))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the main title position
    plt.show()


def denormalize(img):
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (0.2470, 0.2435, 0.2616)
    img = img.astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * channel_stdevs[i]) + channel_means[i]

    return np.transpose(img, (1, 2, 0))


def plot_images(images, predictions, labels, classes):

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Misclassified Images", fontsize=16)
    for i in range(20):
        sub = fig.add_subplot(20 // 5, 5, i + 1)
        image = images[i]
        npimg = denormalize(image.cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        predicted = classes[predictions[i]]
        correct = classes[labels[i]]
        sub.set_title(
            "Correct class: {}\nPredicted class: {}".format(correct, predicted)
        )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the main title position
    plt.show()

def plot_train_images(images, labels, classes):

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Training Images", fontsize=16)
    for i in range(20):
        sub = fig.add_subplot(20 // 5, 5, i + 1)
        image = images[i]
        npimg = denormalize(image.cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        correct = classes[labels[i]]
        sub.set_title(
            "{}".format(correct),
            fontsize=12
        )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the main title position
    plt.show()

def get_cam_visualisation(input_tensor, label, target_layer,model):

    grad_cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)

    targets = [ClassifierOutputTarget(label)]

    grayscale_cam = grad_cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    output = show_cam_on_image(show_transform(input_tensor), grayscale_cam, use_rgb=True, image_weight=0.5)
    return output

def display_gradCam(img,preds,lab,target_layer,classes,model):
    images = []
    predictions = []
    labels = []
    for i in range(20):
        image_ = img[i]
        pred = preds[i]
        label = lab[i]
        image = get_cam_visualisation(image_, pred, target_layer,model)

        target_labels = f"Predicted : {classes[pred]} \n Correct : {classes[label]}"
        images.append(image)
        # predictions.append(target_preds)
        labels.append(target_labels)

    plot_examples(images,labels)

def plot_graphs(train_losses , train_acc , test_losses , test_acc):

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[0, 1].plot(train_acc)
    axs[0, 1].set_title("Training Accuracy")
    axs[1, 0].plot(test_losses)
    axs[1, 0].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

    return

def download_misclassified_images(misclassified_images,ind=100):
  for i in range(ind):
    img = misclassified_images[i]
    imgg = show_transform(img)
    im = Image.fromarray((imgg * 255).astype(np.uint8))
    im.save(f'incorrect/{i}.jpg')

def download_csv(labels,predictions,ind=100):
    clean_label = [label.cpu().item() for label in labels[:ind]]
    clean_prediction = [prediction.cpu().item() for prediction in predictions[:ind]]
    idx = [i for i in range(len(clean_label))]
    my_dict = {
        'index':idx,
        'pred':clean_prediction,
        'truth':clean_label
    }

    return pd.DataFrame(my_dict)

