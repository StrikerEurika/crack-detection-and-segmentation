import cv2
import numpy as np

class CrackMetrics:
    @staticmethod
    def compute_pixel_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict:
        """Computes standard pixel-level segmentation metrics.
        
        Args:
            pred_mask: Binary prediction mask (0 or 255) or boolean array.
            target_mask: Binary ground truth mask (0 or 255) or boolean array.
        """
        p = pred_mask > 127
        t = target_mask > 127
        
        intersection = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        
        pred_sum = p.sum()
        target_sum = t.sum()
        
        precision = float(intersection / pred_sum) if pred_sum > 0 else 0.0
        recall = float(intersection / target_sum) if target_sum > 0 else 0.0
        f1 = float(2 * intersection / (pred_sum + target_sum)) if (pred_sum + target_sum) > 0 else 0.0
        iou = float(intersection / union) if union > 0 else 0.0
        
        return {
            "pixel_precision": round(precision, 4),
            "pixel_recall": round(recall, 4),
            "pixel_f1_dice": round(f1, 4),
            "pixel_iou": round(iou, 4)
        }

    @staticmethod
    def compute_buffered_metrics(pred_skeleton: np.ndarray, target_skeleton: np.ndarray, tolerance: float = 3.0) -> dict:
        """Computes distance-buffered precision, recall, and F1 score for thin structures.
        
        A predicted skeleton pixel is a TP if it lies within `tolerance` pixels of the target skeleton.
        A target skeleton pixel is detected if it lies within `tolerance` pixels of the predicted skeleton.
        
        Args:
            pred_skeleton: 1-pixel wide predicted centerline skeleton (0 or 255).
            target_skeleton: 1-pixel wide ground truth centerline skeleton (0 or 255).
            tolerance: Max Euclidean distance in pixels to count as a match.
        """
        p_coords = np.argwhere(pred_skeleton > 0)
        t_coords = np.argwhere(target_skeleton > 0)
        
        if len(p_coords) == 0 and len(t_coords) == 0:
            return {"buffered_precision": 1.0, "buffered_recall": 1.0, "buffered_f1": 1.0}
        if len(p_coords) == 0:
            return {"buffered_precision": 0.0, "buffered_recall": 0.0, "buffered_f1": 0.0}
        if len(t_coords) == 0:
            return {"buffered_precision": 0.0, "buffered_recall": 0.0, "buffered_f1": 0.0}
            
        # Create distance transforms
        # Distance from every pixel to the nearest target skeleton pixel
        target_dist = cv2.distanceTransform((target_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 5)
        # Distance from every pixel to the nearest predicted skeleton pixel
        pred_dist = cv2.distanceTransform((pred_skeleton == 0).astype(np.uint8), cv2.DIST_L2, 5)
        
        # Precision: fraction of predicted skeleton pixels that are close to target
        p_distances = target_dist[pred_skeleton > 0]
        tp_pred = np.sum(p_distances <= tolerance)
        precision = float(tp_pred / len(p_distances))
        
        # Recall: fraction of target skeleton pixels that are close to prediction
        t_distances = pred_dist[target_skeleton > 0]
        tp_target = np.sum(t_distances <= tolerance)
        recall = float(tp_target / len(t_distances))
        
        # F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        return {
            "buffered_precision": round(precision, 4),
            "buffered_recall": round(recall, 4),
            "buffered_f1": round(f1, 4)
        }
