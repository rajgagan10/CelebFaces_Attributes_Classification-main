from dataclasses import dataclass, field
import os, os.path as osp
from typing import Any, Dict, Optional

import simple_parsing
from simple_parsing.helpers import dict_field


@dataclass
class Hparams:
    """Hyperparameters for the run"""

    # wandb parameters
    wandb_project: str = "classif_celeba"
    wandb_entity: str = "attributes_classification_celeba"       # name of the project
    save_dir: str = osp.join(os.getcwd())                        # directory to save wandb outputs
    weights_path: str = osp.join(os.getcwd(), "weights")

    # train or predict
    train: bool = True
    predict: bool = False

    gpu: int = 0
    fast_dev_run: bool = False
    limit_train_batches: float = 1.0
    val_check_interval: float = 0.5


@dataclass
class TrainParams:
    """Parameters to use for the model"""
    model_name: str = "vit_small_patch16_224"
    pretrained: bool = True
    n_classes: int = 40
    lr: float = 0.00001


@dataclass
class DatasetParams:
    """Parameters for the datamodule"""
    num_workers: int = 2
    root_dataset: Optional[str] = osp.join(os.getcwd(), "assets", "inputs")
    batch_size: int = 1
    input_size: tuple = (224, 224)


@dataclass
class CallBackParams:
    """Parameters for the logging callbacks"""

    nb_image: int = 8
    early_stopping_params: Dict[str, Any] = dict_field(
        dict(
            monitor="val/F1",
            patience=10,
            mode="max",
            verbose=True
        )
    )
    model_checkpoint_params: Dict[str, Any] = dict_field(
        dict(
            monitor="val/F1",
            dirpath=osp.join(os.getcwd(), "weights"),
            filename="best-model",
            mode="max",
            verbose=True
        )
    )


@dataclass
class InferenceParams:
    """Parameters for inference"""
    model_name: str = "vit_small_patch16_224"
    pretrained: bool = True
    n_classes: int = 40
    ckpt_path: Optional[str] = osp.join(os.getcwd(), "weights", "ViTsmall.ckpt")
    output_root: str = osp.join(os.getcwd(), "output")


@dataclass
class SVMParams:
    """Parameters for SVM training"""
    json_file: str = "outputs_stylegan/stylegan3/scores_stylegan3.json"
    np_file: str = "outputs_stylegan/stylegan3/z.npy"
    output_dir: str = "trained_boundaries_z_sg3"
    latent_space_dim: int = 512
    equilibrate: bool = False


@dataclass
class Parameters:
    """Base options."""

    hparams: Hparams = field(default_factory=Hparams)
    data_param: DatasetParams = field(default_factory=DatasetParams)
    callback_param: CallBackParams = field(default_factory=CallBackParams)
    train_param: TrainParams = field(default_factory=TrainParams)
    inference_param: InferenceParams = field(default_factory=InferenceParams)
    svm_params: SVMParams = field(default_factory=SVMParams)

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
