from resnet50.utils.resnet50_predict_cam import is_box_inside_any_focus
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image
from io import BytesIO
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from azure.storage.blob import BlobServiceClient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUSTOM_CATEGORY_NAMES = [
    "__background__",
    "FTU",
    "Sticker",
    "Cable",
    "Screw",
    "Energy box",
    "Heating element"
]

LABEL_MESSAGES = {
    1: {
        "object": "FTU",
        "message": "FTU itself is wrongly placed in general"
    },
    3: {
        "object": "cable (left positioning)",
        "message": "The cable must enter from the left side of the FTU to ensure the original design and proper routing are preserved. | The cable must go straight down for at least 20 cm below the FTU to leave room for the device that connects from underneath."
    },
    4: {
        "object": "screw",
        "message": "Screws should be placed at least 20 cm below the FTU, with the cable running straight and securely fastened above them."
    },
    5: {
        "object": "energy box",
        "message": "The FTU must not be installed directly beside an energy box. Leave sufficient space for other home installations such as breakers, meters, or distribution boards."
    },
    6: {
        "object": "heating element",
        "message": "The FTU should not be installed near heat sources (e.g., radiators or hot water pipes), as heat can degrade the signal quality."
    }
}

def load_fasterrcnn_model(num_classes=91):
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=ftuscannerstorageacc;AccountKey=/ROkqmRf0KP6I/NcpEBLDJn1axLIn01+Cu+ZOTI8dXphbiyKdmnsCiNPHtGksIvixznZB2Fxgg0P+AStlpPb5g==;EndpointSuffix=core.windows.net'
    container_name = 'models'
    blob_name = 'fasterrcnnmodified.pth'

    if not all([connect_str, container_name, blob_name]):
        raise EnvironmentError("One or more Azure storage environment variables are not set.")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)

    tmp_model_path = os.path.join("/tmp", "fasterrcnnmodified.pth")
    with open(tmp_model_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(tmp_model_path, map_location=device))
    model.to(device)
    model.eval()

    return model



def predict_fasterrcnn(model, image_bytes, threshold=0.9):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])[0]

    result = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= threshold:
            result.append({
                "box": box.cpu().numpy().tolist(),
                "label": int(label),
                "score": float(score)
            })
    return result


def box_contains_reddish_pixels(image, box, threshold=0.1):
    xmin, ymin, xmax, ymax = map(int, box)
    region = np.array(image)[ymin:ymax, xmin:xmax]

    if region.size == 0:
        return False

    red_mask = (region[:, :, 0] > 150) & (
        region[:, :, 1] < 100) & (region[:, :, 2] < 100)
    blue_mask = (region[:, :, 2] > 150) & (
        region[:, :, 0] < 100) & (region[:, :, 1] < 100)

    percent_reddish = (red_mask | blue_mask).mean()
    return percent_reddish > threshold


def draw_fasterrcnn_boxes_on_image(image_bytes, predictions, focus_boxes=None, classification_label=None):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Draw border around full image
    border_color = 'green' if classification_label == 'correct' else 'red'
    img_width, img_height = image.size
    border_rect = patches.Rectangle(
        (0, 0), img_width, img_height,
        linewidth=5, edgecolor=border_color, facecolor='none'
    )
    ax.add_patch(border_rect)

    # Identify top predictions
    sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
    high_confidence = [p for p in sorted_preds if p['score'] >= 0.9]
    if len(high_confidence) >= 3:
        top_preds = high_confidence[:3]
    else:
        top_preds = [sorted_preds[0]] if sorted_preds else []

    for pred in predictions:
        xmin, ymin, xmax, ymax = pred["box"]
        box_color = 'lime'
        label_id = pred["label"]
        label_name = CUSTOM_CATEGORY_NAMES[label_id]

        # Determine box color
        if classification_label == "correct" and box_contains_reddish_pixels(image, pred["box"]):
            box_color = 'green'
        elif classification_label == "incorrect" and box_contains_reddish_pixels(image, pred["box"]):
            box_color = 'red'
        elif pred in top_preds:
            box_color = 'blue'

        # Optionally highlight focus
        if classification_label == "incorrect" and focus_boxes:
            if is_box_inside_any_focus(pred["box"], focus_boxes, iou_threshold=0.04):
                box_color = 'red'

        # Draw box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

        # Draw label
        ax.text(xmin, ymin - 5, label_name,
                color=box_color, fontsize=12, weight='bold')

        # Draw PvE message if applicable
        if box_color in ["green", "red"] and label_id in LABEL_MESSAGES:
            message = LABEL_MESSAGES[label_id]["message"]
            wrapped = "\n".join([message[i:i + 60]
                                for i in range(0, len(message), 60)])
            ax.text(xmin, ymax + 5, wrapped,
                    color=box_color, fontsize=8, verticalalignment='top')

    plt.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf
