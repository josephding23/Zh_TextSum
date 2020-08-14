# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "3.0.2"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

# Configurations

from .configuration_bart import BartConfig, MBartConfig

from .data import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    is_sklearn_available,
    squad_convert_examples_to_features,
    xnli_output_modes,
    xnli_processors,
    xnli_tasks_num_labels,
)

# Files and general utilities
from transformers.file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_psutil_available,
    is_py3nvml_available,
    is_tf_available,
    is_torch_available,
    is_torch_tpu_available,
)
from transformers.hf_argparser import HfArgumentParser

# Integrations
from .integrations import is_comet_available, is_tensorboard_available, is_wandb_available

# Model Cards
from transformers.modelcard import ModelCard

# TF 2.0 <=> PyTorch conversion utilities
from transformers.modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model,
)

# Pipelines
from .pipelines import (
    Conversation,
    ConversationalPipeline,
    CsvPipelineDataFormat,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    JsonPipelineDataFormat,
    NerPipeline,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    QuestionAnsweringPipeline,
    SummarizationPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    ZeroShotClassificationPipeline,
    pipeline,
)

# Tokenizers
from transformers.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from transformers.tokenization_bart import BartTokenizer, BartTokenizerFast, MBartTokenizer

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TensorType,
    TokenSpan,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Trainer
from transformers.trainer_utils import EvalPrediction, set_seed
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


if is_sklearn_available():
    from .data import glue_compute_metrics, xnli_compute_metrics


# Modeling
if is_torch_available():

    from transformers.modeling_bart import (
        PretrainedBartModel,
        BartForSequenceClassification,
        BartModel,
        BartForConditionalGeneration,
        BartForQuestionAnswering,
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

    # Optimization
    from transformers.optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    # Trainer
    from transformers.trainer import Trainer, set_seed, torch_distributed_zero_first, EvalPrediction
    from .data.data_collator import (
        default_data_collator,
        DataCollator,
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
    )
    from .data.datasets import (
        GlueDataset,
        TextDataset,
        LineByLineTextDataset,
        GlueDataTrainingArguments,
        SquadDataset,
        SquadDataTrainingArguments,
    )

    # Benchmarks
    from .benchmark.benchmark import PyTorchBenchmark
    from .benchmark.benchmark_args import PyTorchBenchmarkArguments


if not is_torch_available():
    logger.warning(
        "Neither PyTorch nor TensorFlow >= 2.0 have been found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
