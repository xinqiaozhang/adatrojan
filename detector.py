import json
import logging
import joblib
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath, load_ground_truth
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

        # TODO: Update skew parameters per round
        self.model_skew = {
            "__all__": metaparameters["infer_cyber_model_skew"],
        }

        self.input_features = metaparameters["train_input_features"]

    def write_metaparameters(self):
        metaparameters = {
            "infer_cyber_model_skew": self.model_skew["__all__"],
            "train_input_features": self.input_features
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(np.array(sorted(model_repr_dict.keys())).reshape(-1, 1))
        X = []
        y = []

        for model_dirpath in model_path_list:
            model, model_repr, model_class = load_model(
                        join(model_dirpath, "model.pt")
                    )
            model_ground_truth = load_ground_truth(model_dirpath)

            input_grad = self.get_example_gradients(model, join(model_dirpath, 'clean-example-data/'))
            input_grad_norm = np.linalg.norm(input_grad, ord='fro')
            model_class_ohe = enc.transform(np.array([model_class]).reshape(-1, 1)).toarray()[0]
            # concat the one hot encoded model class with the gradient score
            X.append(np.concatenate((model_class_ohe, [input_grad_norm]), axis=0))
            y.append(model_ground_truth)

        X = np.array(X)
        y = np.array(y)

        logging.info("Training Random Forest model...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
        rf.fit(X, y)

        logging.info("Saving model and ohe encoder...")
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(rf, fp)

        joblib.dump(enc, join(self.learned_parameters_dirpath, "ohe_encoder.bin"))

        self.write_metaparameters()
        logging.info("Configuration done!")


    def get_example_gradients(self, model, examples_dirpath):
        """Method to demonstrate how to extract gradients from a round's example data.
        
        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        # Setup scaler
        scaler = StandardScaler()

        scale_params = np.load(self.scale_parameters_filepath)

        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]
        grads = []
        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                # print(f"Processing {examples_dir_entry.path}...")
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()

                feature_vector.requires_grad = True
                model.zero_grad()
                pred = model(feature_vector)
                pred[0, pred.argmax()].backward()

                grad = feature_vector.grad
                grad = grad.detach().numpy().reshape(-1)

                grads.append(grad)
                # print(f"Gradient: {grad}")
        return np.array(grads)

    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        # Setup scaler
        scaler = StandardScaler()

        scale_params = np.load(self.scale_parameters_filepath)

        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]

        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()

                pred = torch.argmax(model(feature_vector).detach()).item()
                ground_truth_filepath = examples_dir_entry.path + ".json"

                with open(ground_truth_filepath, 'r') as ground_truth_file:
                    ground_truth =  ground_truth_file.readline()

                print("Model: {}, Ground Truth: {}, Prediction: {}".format(examples_dir_entry.name, ground_truth, str(pred)))

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """
        # Load model
        model, model_repr, model_class = load_model(model_filepath)

        # Load ohe
        enc = joblib.load(join(self.learned_parameters_dirpath, "ohe_encoder.bin"))

        # Get example gradients
        input_grad = self.get_example_gradients(model, examples_dirpath)
        input_grad_norm = np.linalg.norm(input_grad, ord='fro')
        model_class_ohe = enc.transform(np.array([model_class]).reshape(-1, 1)).toarray()[0]
        # concat the one hot encoded model class with the gradient score
        Xtest = np.concatenate((model_class_ohe, [input_grad_norm]), axis=0).reshape(1, -1)

        with open(self.model_filepath, "rb") as fp:
            classifier: RandomForestRegressor = pickle.load(fp)
        
        probability = classifier.predict(Xtest)[0]
        logging.info("Trojan probability: %s", probability)


        # if model_class[-1].isnumeric():
        #     layer_id = int(model_class[-1])
        # else:
        #     layer_id = int(model_class[-2])
        # logging.info(f"KEY: {model_class} => LAYER ID: {layer_id}")
        # Xtest = [np.hstack((model_repr[f'fc{layer_id}.weight'].flatten(), model_repr[f'fc{layer_id}.bias']))]

        # with open(self.model_filepath, "rb") as fp:
        #     classifier: MLPClassifier = pickle.load(fp)

        # probability = str(classifier.predict_proba(Xtest)[0][1])
        # with open(result_filepath, "w") as fp:
        #     fp.write(probability)

        # logging.info("Trojan probability: %s", probability)
