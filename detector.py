# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, ShuffleSplit

import utils.models
from utils.abstract import AbstractDetector
from utils.models import load_model, load_ground_truth

import torch
import torchvision
import skimage.io
from collections import defaultdict
from tqdm import tqdm


def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - (0.5 * w)), (y_c - (0.5 * h)), (x_c + (0.5 * w)), (y_c + (0.5 * h))]
    return torch.stack(b, dim=-1)


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = os.path.join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = os.path.join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = os.path.join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = os.path.join(self.learned_parameters_dirpath, "layer_transform.bin")

        self.input_features = metaparameters["train_input_features"]
        self.weight_params = {
            "rso_seed": metaparameters["train_weight_rso_seed"],
            "mean": metaparameters["train_weight_params_mean"],
            "std": metaparameters["train_weight_params_std"],
        }
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_regressor_param_n_estimators"
            ],
            "criterion": metaparameters[
                "train_random_forest_regressor_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_regressor_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_regressor_param_min_samples_split"
            ],
            "min_samples_leaf": metaparameters[
                "train_random_forest_regressor_param_min_samples_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_regressor_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_regressor_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_regressor_param_min_impurity_decrease"
            ],
        }

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_weight_rso_seed": self.weight_params["rso_seed"],
            "train_weight_params_mean": self.weight_params["mean"],
            "train_weight_params_std": self.weight_params["std"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))

    def cross_val_report(self, X, y, scoring='roc_auc'):
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        SVM_model = svm.SVC(kernel='linear', C=1, random_state=42)
        scores = cross_val_score(SVM_model, X, y, cv=cv, scoring=scoring)
        print("SVM: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        RF_model = RandomForestClassifier(max_depth=5, n_estimators=200)
        scores = cross_val_score(RF_model, X, y, cv=cv, scoring=scoring)
        print("RF: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    def svd_in_dim(self, weight, dim):
        u, s, v = torch.svd(weight.reshape(weight.shape[dim], -1), compute_uv=False)
        return s
    
    def get_features(self, model):
        svd_list = []
        for name, param in model.named_parameters():
            weight = param.data
            numdims = len(weight.shape)
            if numdims > 1:
                if numdims > 3:
                    for mydim in range(numdims - 2):
                        s = self.svd_in_dim(weight,mydim).detach().cpu().numpy()
                        svd_list.extend([s[0],s[1],s[2]])
                else:
                    s = self.svd_in_dim(weight,0).detach().cpu().numpy()
                    svd_list.extend([s[0],s[1],s[2]])
        
        return np.asarray(svd_list)

    def load_models_dirpath(self, models_dirpath, training):
        X = defaultdict(list)
        Y = defaultdict(list)

        for model_path in tqdm(models_dirpath):
            model, model_repr, model_class = load_model(os.path.join(model_path, "model.pt"))
            model_ground_truth = load_ground_truth(model_path)

            feats = self.get_features(model)
            X[model_class].append(feats)
            if training:
                Y[model_class].append(model_ground_truth)

        return X, Y if training else X

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """

        self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        X, Y = self.load_models_dirpath(model_path_list, training=True)
        for model_class in X.keys():
            X[model_class] = np.asarray(X[model_class])
            Y[model_class] = np.asarray(Y[model_class])
            
            # standard scaling
            scaler = StandardScaler()
            X[model_class] = scaler.fit_transform(X[model_class])

            clf = svm.SVC(kernel='linear', C=1, random_state=42)
            rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='roc_auc', min_features_to_select=10)
            rfecv.fit(X[model_class], Y[model_class])

            ## Get cv scores
            print("Optimal number of features : %d" % rfecv.n_features_)
            print(model_class)
            self.cross_val_report(X[model_class][:, rfecv.support_], Y[model_class], scoring='accuracy')
            
            newfeats = X[model_class][:, rfecv.support_]

            ## Fit model
            clf = svm.SVC(kernel='linear', C=1, random_state=42, probability=True)
            clf.fit(newfeats, Y[model_class])
            calibrator = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
            calibrator.fit(newfeats, Y[model_class])
            print(calibrator.score(newfeats, Y[model_class]))

            # save model, scaler, rfecv
            logging.info("Saving Model, Scaler, RFECV for %s...", model_class)
            with open('{}/model_{}.pkl'.format(self.learned_parameters_dirpath, model_class), 'wb') as f:
                pickle.dump(calibrator, f)
            with open('{}/scaler_{}.pkl'.format(self.learned_parameters_dirpath, model_class), 'wb') as f:
                pickle.dump(scaler, f)
            with open('{}/rfecv_{}.pkl'.format(self.learned_parameters_dirpath, model_class), 'wb') as f:
                pickle.dump(rfecv, f)

        self.write_metaparameters()
        logging.info("Configuration done!")


    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        # move the model to the GPU in eval mode
        model.to(device)
        model.eval()

        # Augmentation transformations
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        logging.info("Evaluating the model on the clean example images.")
        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
                # load the example image
                img = skimage.io.imread(examples_dir_entry.path)

                # convert the image to a tensor
                # should be uint8 type, the conversion to float is handled later
                image = torch.as_tensor(img)

                # move channels first
                image = image.permute((2, 0, 1))

                # convert to float (which normalizes the values)
                image = augmentation_transforms(image)
                image = image.to(device)

                # Convert to NCHW
                image = image.unsqueeze(0)

                # inference
                outputs = model(image)
                # handle multiple output formats for different model types
                if 'DetrObjectDetectionOutput' in outputs.__class__.__name__:
                    # DETR doesn't need to unpack the batch dimension
                    boxes = outputs.pred_boxes.cpu().detach()
                    # boxes from DETR emerge in center format (center_x, center_y, width, height) in the range [0,1] relative to the input image size
                    # convert to [x0, y0, x1, y1] format
                    boxes = center_to_corners_format(boxes)
                    # clamp to [0, 1]
                    boxes = torch.clamp(boxes, min=0, max=1)
                    # and from relative [0, 1] to absolute [0, height] coordinates
                    img_h = img.shape[0] * torch.ones(1)  # 1 because we only have 1 image in the batch
                    img_w = img.shape[1] * torch.ones(1)
                    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                    boxes = boxes * scale_fct[:, None, :]

                    # unpack the logits to get scores and labels
                    logits = outputs.logits.cpu().detach()
                    prob = torch.nn.functional.softmax(logits, -1)
                    scores, labels = prob[..., :-1].max(-1)

                    boxes = boxes.numpy()
                    scores = scores.numpy()
                    labels = labels.numpy()

                    # all 3 items have a batch size of 1 in the front, so unpack it
                    boxes = boxes[0,]
                    scores = scores[0,]
                    labels = labels[0,]
                else:
                    # unpack the batch dimension
                    outputs = outputs[0]  # unpack the batch size of 1
                    # for SSD and FasterRCNN outputs are a list of dict.
                    # each boxes is in corners format (x_0, y_0, x_1, y_1) with coordinates sized according to the input image

                    boxes = outputs['boxes'].cpu().detach().numpy()
                    scores = outputs['scores'].cpu().detach().numpy()
                    labels = outputs['labels'].cpu().detach().numpy()

                # wrap the network outputs into a list of annotations
                pred = utils.models.wrap_network_prediction(boxes, labels)

                # logging.info('example img filepath = {}, Pred: {}'.format(examples_dir_entry.name, pred))

                ground_truth_filepath = examples_dir_entry.path.replace('.png','.json')

                with open(ground_truth_filepath, mode='r', encoding='utf-8') as f:
                    ground_truth = jsonpickle.decode(f.read())

                logging.info("Model predicted {} boxes, Ground Truth has {} boxes.".format(len(pred), len(ground_truth)))
                # logging.info("Model: {}, Ground Truth: {}".format(examples_dir_entry.name, ground_truth))


    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        # load the model
        model, model_repr, model_class = load_model(model_filepath)

        feats = self.get_features(model)
        X = [feats]

        # load the scaler, rfecv, model
        with open('{}/scaler_{}.pkl'.format(self.learned_parameters_dirpath, model_class), 'rb') as f:
            scaler = pickle.load(f)
        with open('{}/rfecv_{}.pkl'.format(self.learned_parameters_dirpath, model_class), 'rb') as f:
            rfecv = pickle.load(f)
        with open('{}/model_{}.pkl'.format(self.learned_parameters_dirpath, model_class), 'rb') as f:
            classifier = pickle.load(f)
        
        # standard scaling
        X = scaler.transform(X)
        # feature selection
        X = X[:, rfecv.support_]
        # inference
        probability = classifier.predict_proba(X)[0][1]
        # clip the probability to reasonable values
        probability = np.clip(probability, a_min=0.01, a_max=0.99)

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))

