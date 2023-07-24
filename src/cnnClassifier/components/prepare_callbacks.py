import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig
from pathlib import Path


class PrepareCallback:
    def __init__(self, config):
        self.config = config
        
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    def _create_ckpt_callbacks(self):
        checkpoint_dir = os.path.join(
            self.config.checkpoint_root_dir,
            "model_checkpoint",
        )
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            # other parameters...
        )

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks(),  # Don't call the function here, just pass the reference
            self._create_ckpt_callbacks(),  # Don't call the function here, just pass the reference
        ]
