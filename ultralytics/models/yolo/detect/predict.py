# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import torch
import copy


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    def postprocess_original(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""        
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        
        

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred, appearance_features=None, appearance_feature_map=None))
    
        return results

    def postprocess(self, preds, img, orig_imgs, appearance_feature_layer=None):
        """Post-processes predictions and returns a list of Results objects."""
        if appearance_feature_layer is None:
            return self.postprocess_original(preds, img, orig_imgs)
        
        if isinstance(preds, dict):
            feature_map = preds["feature_map"]
        else:
            feature_map = preds[-1] 
        
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )
        
        results = []
        
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred_boxes = copy.deepcopy(pred)
            
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            
            features, feature_map = self.extract_appearance_features(feature_map, pred_boxes, appearance_feature_layer, img)
            
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred, appearance_features=features, appearance_feature_map=feature_map))      
        return results

    def extract_appearance_features(self, feature_map, preds, appearance_feature_layer, img):
        feature_map = feature_map[appearance_feature_layer][0,:, :, :]  # (48, 368, 640)
        reshaped_feature_map = feature_map.permute(1, 2, 0)  # (368, 640, 48)
        
        feature_dim = reshaped_feature_map.shape[-1] # cmap

        preds[:, :4] = ops.scale_boxes( img.shape[2:], preds[:, :4], reshaped_feature_map.shape)

        boxes = preds[:, :4].long().cpu().numpy()
        
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