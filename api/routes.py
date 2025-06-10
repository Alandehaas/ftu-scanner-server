from faster_rcnn.utils.fasterrcnn_predict_draw import draw_fasterrcnn_boxes_on_image
from resnet50.utils.resnet50_predict_cam import (
    get_focus_box_from_cam, is_box_inside_focus, draw_resnet_cam
)
from flask import Blueprint, request, jsonify, send_file
from io import BytesIO
import base64

from faster_rcnn.utils.fasterrcnn_predict_draw import load_fasterrcnn_model, predict_fasterrcnn
from resnet50.utils.resnet50_predict_cam import load_resnet_model, predict_resnet, draw_resnet_cam, is_box_inside_any_focus

routes = Blueprint('routes', __name__)

fasterrcnn_model = load_fasterrcnn_model(num_classes=7)
resnet_model = load_resnet_model()

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


# Faster R-CNN Route
@routes.route('/fasterrcnn', methods=['POST'])
def predict_fasterrcnn_route():
    try:
        image_bytes = get_image_bytes(request)
        if isinstance(image_bytes, dict):
            return jsonify(image_bytes), 400

        predictions = predict_fasterrcnn(fasterrcnn_model, image_bytes)

        if request.args.get("return_image", "").strip().lower() == "true":
            image_with_boxes = draw_fasterrcnn_boxes_on_image(
                image_bytes, predictions)
            return send_file(image_with_boxes, mimetype='image/png')

        return jsonify({"prediction": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ResNet-50 + Grad-CAM Route
@routes.route('/resnet50', methods=['POST'])
def predict_resnet_route():
    try:
        image_bytes = get_image_bytes(request)
        if isinstance(image_bytes, dict):
            return jsonify(image_bytes), 400

        prediction_label = predict_resnet(resnet_model, image_bytes)

        if request.args.get("return_image", "").strip().lower() == "true":
            image_with_overlay, _ = draw_resnet_cam(
                resnet_model, image_bytes, prediction_label)
            return send_file(image_with_overlay, mimetype='image/png')

        return jsonify({"prediction": prediction_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Utility Function (Shared)
def get_image_bytes(req):
    if "file" in req.files:
        file = req.files["file"]
        if file.filename == "":
            return {"error": "No file selected"}
        return file.read()
    elif req.is_json and "file" in req.json:
        try:
            return base64.b64decode(req.json["file"])
        except Exception:
            return {"error": "Invalid base64 image data"}
    else:
        return {"error": "No image data provided"}


@routes.route('/fused', methods=['POST'])
def fused_prediction_route():
    try:
        image_bytes = get_image_bytes(request)
        if isinstance(image_bytes, dict):
            return jsonify(image_bytes), 400

        predictions = predict_fasterrcnn(fasterrcnn_model, image_bytes)

        classification_label = predict_resnet(resnet_model, image_bytes)

        heatmap_image, focus_boxes = draw_resnet_cam(
            resnet_model, image_bytes, classification_label)

        in_focus = [p for p in predictions if is_box_inside_any_focus(
            p["box"], focus_boxes, iou_threshold=0.04)]

        if request.args.get("return_image", "").strip().lower() == "true":
            heatmap_image, _ = draw_resnet_cam(
                resnet_model, image_bytes, classification_label)

            final_image = draw_fasterrcnn_boxes_on_image(
                heatmap_image.getvalue(),
                predictions,
                focus_boxes=focus_boxes,
                classification_label=classification_label
            )
            return send_file(final_image, mimetype='image/png')

        return jsonify({
            "classification": classification_label,
            "heatmap_focus_box": focus_boxes,
            "detections_in_focus": in_focus,
            "total_detections": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@routes.route('/result', methods=['POST'])
def fused_label_check_route():
    try:
        image_bytes = get_image_bytes(request)
        if isinstance(image_bytes, dict):
            return jsonify(image_bytes), 400

        # Run classification + detection
        predictions = predict_fasterrcnn(fasterrcnn_model, image_bytes)
        classification_label = predict_resnet(resnet_model, image_bytes)
        heatmap_image, focus_boxes = draw_resnet_cam(
            resnet_model, image_bytes, classification_label)

        # Get detections inside Grad-CAM focus regions
        in_focus = [
            p for p in predictions if is_box_inside_any_focus(p["box"], focus_boxes, iou_threshold=0.04)
        ]

        # No focused objects found
        if not in_focus:
            return jsonify({
                "message": "No objects found in the area of interest"
            })

        # Take the first detection in focus (or process all)
        detected_label = in_focus[0]["label"]

        label_info = LABEL_MESSAGES.get(detected_label, {
            "object": f"label {detected_label}",
            "message": "No specific message defined for this object."
        })

        return jsonify({
            "object": label_info["object"],
            "message": label_info["message"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@routes.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200
@routes.route('/status', methods=['GET'])
def status_check():
    return jsonify({
        "fasterrcnn": "loaded",
        "resnet50": "loaded"
    }), 200