import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Union, List
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from src.config import OUTPUTS_DIR, CLASSES
from src.utils import setup_logger

logger = setup_logger(__name__)

class Evaluator:
    def __init__(self, output_dir: Union[str, Path] = OUTPUTS_DIR):
        """
        Initializes the Evaluator.
        
        Args:
            output_dir (Union[str, Path]): Directory where evaluation artifacts will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_confusion_matrix(self, y_true: List[int], y_pred: List[int], save_path: str = "confusion_matrix.png"):
        """Generates and saves the confusion matrix."""
        logger.info("Generating Confusion Matrix...")
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            full_path = self.output_dir / save_path
            plt.savefig(full_path)
            plt.close()
            logger.info(f"Saved Confusion Matrix to {full_path}")
        except Exception as e:
            logger.error(f"Failed to generate Confusion Matrix: {str(e)}")
        
    def generate_roc_curve(self, y_true: List[int], y_prob: List[float], save_path: str = "roc_curve.png"):
        """Generates and saves the ROC curve."""
        logger.info("Generating ROC Curve...")
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.tight_layout()
            
            full_path = self.output_dir / save_path
            plt.savefig(full_path)
            plt.close()
            logger.info(f"Saved ROC Curve to {full_path}")
        except Exception as e:
            logger.error(f"Failed to generate ROC Curve: {str(e)}")

    def generate_pr_curve(self, y_true: List[int], y_prob: List[float], save_path: str = "pr_curve.png"):
        """Generates and saves the Precision-Recall curve."""
        logger.info("Generating Precision-Recall Curve...")
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.tight_layout()
            
            full_path = self.output_dir / save_path
            plt.savefig(full_path)
            plt.close()
            logger.info(f"Saved PR Curve to {full_path}")
        except Exception as e:
            logger.error(f"Failed to generate PR Curve: {str(e)}")

    def evaluate(self, y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
        """
        Runs full evaluation suite.
        
        Args:
            y_true: Ground truth labels (0 or 1).
            y_pred: Predicted labels (0 or 1).
            y_prob: Predicted probabilities for the positive class.
            
        Returns:
            Dict[str, float]: Dictionary containing core evaluation metrics.
        """
        logger.info("Starting Full Evaluation...")
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        }
        
        try:
            report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
            
            # Save report to text file
            report_path = self.output_dir / "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
                f.write("\n\nMetrics Summary:\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.4f}\n")
                    
            logger.info(f"Saved Classification Report to {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate classification report: {str(e)}")
        
        # Generate graphs
        self.generate_confusion_matrix(y_true, y_pred)
        self.generate_roc_curve(y_true, y_prob)
        self.generate_pr_curve(y_true, y_prob)
        
        logger.info("Evaluation complete.")
        return metrics
