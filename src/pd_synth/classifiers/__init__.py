"""Dataset-agnostic classifier model and train/eval loop."""

from pd_synth.classifiers.simple_cnn import SimpleCNN
from pd_synth.classifiers.train import TrainResult, evaluate_classifier, train_classifier

__all__ = ["SimpleCNN", "TrainResult", "evaluate_classifier", "train_classifier"]
