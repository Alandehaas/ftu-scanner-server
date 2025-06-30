import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import os
from azure.storage.blob import BlobServiceClient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['correct', 'incorrect']
NUM_CLASSES = len(CLASS_NAMES)

def load_resnet_model():
    # Relative to where main.py is executed from (kpn/)
    model_path = "resnet50/models/resnet50.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict_resnet(model, image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform_image(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]


def draw_resnet_cam(model, image_bytes, prediction_label):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform_image(image).to(DEVICE)

    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1].conv3
    target_layer.register_forward_hook(save_activation)
    target_layer.register_full_backward_hook(save_gradient)

    output = model(input_tensor)
    target_class = CLASS_NAMES.index(prediction_label)
    target = output[0, target_class]

    model.zero_grad()
    target.backward(retain_graph=True)

    grads = gradients[0].detach()
    acts = activations[0].detach()

    B, C, H, W = grads.shape
    alpha_num = grads.pow(2)
    alpha_denom = grads.pow(2) * 2 + acts * grads.pow(3)
    alpha_denom = torch.where(
        alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    weights = (alphas * torch.relu(grads)).sum(dim=(2, 3))
    saliency_map = (weights[:, :, None, None] * acts).sum(dim=1).squeeze()
    saliency_map = torch.relu(saliency_map)
    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()

    heatmap = saliency_map.cpu().numpy()
    heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
    focus_boxes = get_all_focus_boxes(heatmap_resized)

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
    original_img = np.array(image)
    superimposed_img = cv2.addWeighted(
        original_img, 0.6, heatmap_colored, 0.4, 0)

    result_img = Image.fromarray(superimposed_img)
    buffer = BytesIO()
    result_img.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer, focus_boxes


def get_all_focus_boxes(heatmap_np, threshold_percentile=90):
    threshold_value = np.percentile(heatmap_np, threshold_percentile)
    mask = np.uint8(heatmap_np >= threshold_value) * 255
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h])
    return boxes


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0
    return intersection / union


def is_box_inside_any_focus(pred_box, focus_boxes, iou_threshold=0.1):
    for focus_box in focus_boxes:
        iou = compute_iou(pred_box, focus_box)
        if iou >= iou_threshold:
            return True
    return False


def is_box_inside_focus(pred_box, focus_box):
    if focus_box is None:
        return False
    x_center = (pred_box[0] + pred_box[2]) / 2
    y_center = (pred_box[1] + pred_box[3]) / 2
    fxmin, fymin, fxmax, fymax = focus_box
    return fxmin <= x_center <= fxmax and fymin <= y_center <= fymax


def get_focus_box_from_cam(model, image_bytes, prediction_label):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform_image(image).to(DEVICE)

    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1].conv3
    target_layer.register_forward_hook(save_activation)
    target_layer.register_full_backward_hook(save_gradient)

    output = model(input_tensor)
    target_class = CLASS_NAMES.index(prediction_label)
    target = output[0, target_class]

    model.zero_grad()
    target.backward(retain_graph=True)

    grads = gradients[0].detach()
    acts = activations[0].detach()

    B, C, H, W = grads.shape
    alpha_num = grads.pow(2)
    alpha_denom = grads.pow(2) * 2 + acts * grads.pow(3)
    alpha_denom = torch.where(
        alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    weights = (alphas * torch.relu(grads)).sum(dim=(2, 3))
    saliency_map = (weights[:, :, None, None] * acts).sum(dim=1).squeeze()
    saliency_map = torch.relu(saliency_map)
    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()

    heatmap = saliency_map.cpu().numpy()
    heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
    return get_all_focus_boxes(heatmap_resized)
