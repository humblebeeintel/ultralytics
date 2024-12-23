# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
import cv2
import copy


class DetectionPredictor(BasePredictor):

    def postprocess_original(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds, # preds[1] for tensorrt
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(
                img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path,
                           names=self.model.names, boxes=pred, appearance_features=None, appearance_feature_map=None))
        return results

    def postprocess(self, preds, img, orig_imgs, appearance_feature_layer=None):
        """
        Postprocesses predictions and returns a list of Results objects.
        Additionally, it returns low-level appearance features of the detected objects.
        preds has 3 elements: yolo output (1,80+4,N), 3 anchor layer grids, first layer feature map.

        Args:
            preds (list): Predictions from the model.
            img (torch.Tensor): Input image in [batchsize, channels, height, width] format.
            orig_imgs (torch.Tensor or list): Original images.
            appearance_feature_layer (int, optional): The layer to extract appearance features from.

        Returns:
            list: List of Results objects with detections and optional appearance features.
        """
        if appearance_feature_layer is None:
            return self.postprocess_original(preds, img, orig_imgs)

        # Extract the feature map if appearance_feature_layer is not None
        # appearance_feature_layer can be layer1 ... layer22

        features_dict = copy.deepcopy(preds[2])

        # Apply Non-Max Suppression (NMS)
        preds = ops.non_max_suppression(
            preds, # preds[1] for tensorrt 
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes
        )

        preds_map = copy.deepcopy(preds)

        results = []

        for i, pred in enumerate(preds):  # each i is an image
            orig_img = orig_imgs[i] if isinstance(
                orig_imgs, list) else orig_imgs

            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], orig_img.shape)

            # Feature map extraction for appearance-based tracking
            # bounding boxes are in img.shape format, so we need to scale them to feature map resolution
            if appearance_feature_layer is None:
                features = None
            
            else:
                features, feature_map = self.extract_appearance_features(features_dict, preds_map, 
                                                            appearance_feature_layer, img)


            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path

            results.append(Results(orig_img=orig_img, path=img_path,
                           names=self.model.names, boxes=pred, appearance_features=features, 
                           appearance_feature_map=feature_map))

        return results

    def extract_appearance_features(self, feature_map, preds, appearance_feature_layer, img):
        feature_map = feature_map[appearance_feature_layer][0,:, :, :]  # (48, 368, 640)
        reshaped_feature_map = feature_map.permute(1, 2, 0)  # (368, 640, 48)
        
        feature_dim = reshaped_feature_map.shape[-1] # cmap

        preds[0][:, :4] = ops.scale_boxes(
            img.shape[2:], preds[0][:, :4], reshaped_feature_map.shape)
        boxes = preds[0][:, :4].long().cpu().numpy()
        features_normalized = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            
            # (48, height, width)
            extracted_feature = feature_map[:, y_min:y_max, x_min:x_max]

            if 0 not in extracted_feature.shape:
                feature_mean = torch.mean(
                    extracted_feature, dim=(1, 2))  # (48,)
                normalized_feature = feature_mean / \
                    feature_mean.norm(p=2, dim=0, keepdim=True)
            else:
                normalized_feature = torch.ones(
                    feature_dim, dtype=torch.float32, device=reshaped_feature_map.device)

            features_normalized.append(normalized_feature)

        features = torch.stack(
            features_normalized, dim=0) if features_normalized else torch.tensor([])
        
        return features, feature_map


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
