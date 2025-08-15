# training/logger.py
import os
from torch.utils.tensorboard import SummaryWriter
import csv
from datetime import datetime
from utils.path import LOG_DIR

tranning_log_dir = LOG_DIR

class Logger:
    def __init__(self, tranning_log_dir):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_dir = os.path.join(tranning_log_dir, timestamp)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.csv_file = os.path.join(self.tensorboard_dir, "training_log.csv")
        self.csv_initialized = False

    def log_metrics(self, metrics_dict, epoch):
        """
        metrics_dict: dict, e.g. {"train_loss": 0.1, "val_loss": 0.2, ...}
        """
        # TensorBoard logging
        for key, value in metrics_dict.items():
            self.writer.add_scalar(key, value, epoch)

        # CSV logging
        if not self.csv_initialized:
            with open(self.csv_file, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["epoch"] + list(metrics_dict.keys()))
                writer.writeheader()
            self.csv_initialized = True

        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + list(metrics_dict.keys()))
            row = {"epoch": epoch, **metrics_dict}
            writer.writerow(row)

    def close(self):
        self.writer.close()
