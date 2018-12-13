import logging
import os
import shutil
import time
import re
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Set

import torch
import torch.optim.lr_scheduler
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import gather
from tensorboardX import SummaryWriter

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb, scatter_kwargs
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer
import datetime
from models.dialog_qa_ctx import DialogQA

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class TwoPhaseTrainer(Trainer):
    """
        Note: this is kind of brittle, only to be used with dialog_qa_ctx models
    """
    def __init__(self, model: DialogQA,
                 phase1_trainer: Trainer,
                 phase2_trainer: Trainer) -> None:
        self._model = model
        self._phase1_trainer = phase1_trainer
        self._phase2_trainer = phase2_trainer

    def train(self):
        try:
            epoch_counter, validation_metric_per_epoch = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")


        logger.info('Training base QA')
        self._model.train_base_qa()  # train base qa model
        self._phase1_trainer.train()
        logger.info('Training coref module')
        self._model.train_coref_module()
        self._phase2_trainer.train()

    @classmethod
    def from_params(cls,  # type: ignore
                    model: DialogQA,
                    serialization_dir: str,
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    params: Params,
                    validation_iterator: DataIterator = None) -> 'Trainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_p1_epochs = params.pop_int("num_phase1_epochs", 20)
        num_p2_epochs = params.pop_int("num_phase2_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        p1_optim_params = params.pop("phase1_optimizer")
        p2_optim_params = params.pop("phase2_optimizer")


        parameters = [(n, p) for n, p in model.named_parameters()
                             if p.requires_grad]

        phase1_optimizer = Optimizer.from_params(parameters, p1_optim_params)
        phase2_optimizer = Optimizer.from_params(parameters, p2_optim_params)


        if lr_scheduler_params:
            phase1_scheduler = LearningRateScheduler.from_params(phase1_optimizer, lr_scheduler_params)
            phase2_scheduler = None  # no lr scheduling for now
        else:
            phase1_scheduler = None
            phase2_scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
            "keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)

        params.assert_empty(cls.__name__)

        phase1_trainer = Trainer(model, phase1_optimizer, iterator,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_p1_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=phase1_scheduler,
                   num_serialized_models_to_keep=num_serialized_models_to_keep,
                   keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate)

        phase2_trainer = Trainer(model, phase2_optimizer, iterator,
                                 train_data, validation_data,
                                 patience=patience,
                                 validation_metric=validation_metric,
                                 validation_iterator=validation_iterator,
                                 shuffle=shuffle,
                                 num_epochs=num_p2_epochs,
                                 serialization_dir=serialization_dir,
                                 cuda_device=cuda_device,
                                 grad_norm=grad_norm,
                                 grad_clipping=grad_clipping,
                                 learning_rate_scheduler=phase2_scheduler,
                                 num_serialized_models_to_keep=num_serialized_models_to_keep,
                                 keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                                 model_save_interval=model_save_interval,
                                 summary_interval=summary_interval,
                                 histogram_interval=histogram_interval,
                                 should_log_parameter_statistics=should_log_parameter_statistics,
                                 should_log_learning_rate=should_log_learning_rate)

        return cls(model, phase1_trainer, phase2_trainer)

Trainer.register("two_phase_trainer")(TwoPhaseTrainer)