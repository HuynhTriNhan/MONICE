"""Constants for dice-ml package."""
from enum import Enum


class RobustnessType(Enum):
    DICE_SORENSEN = "dice-sorensen"
    GATED_DICE_SORENSEN = "gated-dice-sorensen"
    GAUSSIAN_KERNEL = "gaussian-kernel"
    BINNED_GAUSSIAN_KERNEL = "binned-gaussian-kernel"


class BackEndTypes:
    Sklearn = 'sklearn'
    Tensorflow1 = 'TF1'
    Tensorflow2 = 'TF2'
    Pytorch = 'PYT'
    DPPytorch = 'DP_PYT'

    ALL = [Sklearn, Tensorflow1, Tensorflow2, Pytorch, DPPytorch]


class SamplingStrategy:
    Random = 'random'
    Genetic = 'genetic'
    KdTree = 'kdtree'
    Gradient = 'gradient'


class ModelTypes:
    Classifier = 'classifier'
    Regressor = 'regressor'

    ALL = [Classifier, Regressor]


class _SchemaVersions:
    V1 = '1.0'
    V2 = '2.0'
    CURRENT_VERSION = V2

    ALL_VERSIONS = [V1, V2]


class _PostHocSparsityTypes:
    LINEAR = 'linear'
    BINARY = 'binary'

    ALL = [LINEAR, BINARY]
