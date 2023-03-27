# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

#0828 Try to draw the distribute of different models. Look good result!!!! --> not consistant, hard to find some situation

import os
from os import listdir, makedirs
import numpy as np
import cv2
import pickle
import torch
import torchvision
# from matplotlib import pyplot as plt
import json
import jsonschema
from numpy import arange
import logging
import warnings 
import pdb
import collections
import pandas as pd
# from scipy.special import softmax
# from scipy.stats import norm
import math
from os.path import join, exists, basename
from utils.models import create_layer_map, load_model, \
    load_models_dirpath, load_ground_truth
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

def label_shift(model, data, svec, epsilon):
    '''
    takes the singular vector for each class as a Trojan patch candidate, adds each to a batch of true data
    points and looks at the shift in classification for various scalings.
    For trojaned models these are very effective and switching classifications to the target class,
    clean models are less responsive.  The response for epsilon < 5 and epsilon >20 are particularly distinguishing
    '''
    C = model(data[0:1]).shape[1]
    n = len(data)
    pct_shift = torch.zeros(C,C,len(epsilon)).to(device)
    for i,s in enumerate(svec):
        v = s.reshape(3,224,224).to(device)
        for j,e in enumerate(epsilon):
            pred = model(data + e*v).argmax(dim=1)
            unique, counts = torch.unique(pred, return_counts=True)
            pct_shift[i, unique.long(), j] = counts.float() / n

    return pct_shift

def get_example_gradients(model, examples_dirpath):
    """Method to demonstrate how to extract gradients from a round's example data.

    Args:
        model: the pytorch model
        examples_dirpath: the directory path for the round example data
    """
    # Setup scaler
    scaler = StandardScaler()

    # scale_params = np.load(scale_parameters_filepath)

    # scaler.mean_ = scale_params[0]
    # scaler.scale_ = scale_params[1]
    grads = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])
    # Inference on models
    for examples_dir_entry in os.scandir(examples_dirpath):
        if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
            # print(f"Processing {examples_dir_entry.path}...")
            fn = examples_dir_entry.path 
            img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # convert the image to a tensor
            # should be uint8 type, the conversion to float is handled later
            image = torch.as_tensor(img)

            # move channels first
            image = image.permute((2, 0, 1))

            # convert to float (which normalizes the values)
            image = augmentation_transforms(image)
            image = image.to(device)
            model = model.to(device)
            

            # Convert to NCHW
            image = image.unsqueeze(0)
            image.requires_grad = True
            # inference
            model.zero_grad()
            pred = model(image)
            pred[0, pred.argmax()].backward()

            # pdb.set_trace()
            grad = image.grad
            grad = grad.cpu().detach().numpy().reshape(-1)

            grads.append(grad)
            # feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
            # feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()
            # feature_vector.requires_grad = True

            # ground_truth_filepath = examples_dir_entry.path + ".json"

            # with open(ground_truth_filepath, 'r') as ground_truth_file:
            #     ground_truth =  ground_truth_file.readline()

            # model.zero_grad()
            # loss_fn = nn.CrossEntropyLoss()
            # loss = loss_fn(model(feature_vector), torch.tensor([int(ground_truth)]))
            # loss.backward()

            # model.zero_grad()
            # pred = model(feature_vector)
            # pred[0, pred.argmax()].backward()

            # grad = feature_vector.grad
            # grad = grad.detach().numpy().reshape(-1)

            # grads.append(grad)
            # print(f"Gradient: {grad}")
    return np.array(grads)

def example_trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, parameters_dirpath, parameter1, parameter2, example_img_format='jpg'):
    logging.info('model_filepath = {}'.format(model_filepath))
    logging.info('result_filepath = {}'.format(result_filepath))
    logging.info('scratch_dirpath = {}'.format(scratch_dirpath))
    logging.info('examples_dirpath = {}'.format(examples_dirpath))
    logging.info('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    logging.info('Using parameters_dirpath = {}'.format(parameters_dirpath))
    logging.info('Using parameter1 = {}'.format(parameter1))
    logging.info('Using parameter2 = {}'.format(parameter2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info("Using compute device: {}".format(device))
    model, model_repr, model_class = load_model(model_filepath)
                    #     join(model_filepath, "model.pt")
                    # )
    # model_ground_truth = load_ground_truth(model_dirpath)

    # synthetic_grad_norm = self.get_synthetic_gradients(model)
    
    # input_grad = get_example_gradients(model, join(model_filepath, 'clean-example-data/'))
    input_grad = get_example_gradients(model, examples_dirpath)
    input_grad_norm = np.linalg.norm(input_grad, ord='fro')

    layers = [n for n, _ in model.named_modules()]
    weights = model_repr[layers[-1]+'.weight'].T
    s = np.linalg.svd(weights, compute_uv=False)[0:1]
    # X.append(np.concatenate(([input_grad_norm], s), axis=0))
    Xtest = np.concatenate(([input_grad_norm], s), axis=0).reshape(1, -1)
    classifier_dirpath = join(parameters_dirpath, "model.bin")
    with open(classifier_dirpath, "rb") as fp:
        classifier: CalibratedClassifierCV = pickle.load(fp)
        
    probability = classifier.predict(Xtest)[0]
    # limit probability to [0, 1]
    probability = max(0.0, min(1.0, probability))
    # load the model and move it to the GPU
    # model = torch.load(model_filepath)
    # model.to(device)
    # model.eval()

    # Augmentation transformations
    # augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

    # Inference the example images in data
    # fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
    # fns.sort()  # ensure file ordering
    # if len(fns) > 5: fns = fns[0:5]  # limit to 5 images

    
    # trojan_probability = np.random.rand()
    logging.info("Trojan probability: %s", probability)


    with open(result_filepath, "w") as fp:
        fp.write(str(probability))



def configure(output_parameters_dirpath,
              configure_models_dirpath,
              parameter3):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(output_parameters_dirpath):
            makedirs(output_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(configure_models_dirpath, model) for model in listdir(configure_models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        # model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)
        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc.fit(np.array(sorted(model_repr_dict.keys())).reshape(-1, 1))
        X = []
        y = []

        logging.info("Generating training data...")
        for model_dirpath in model_path_list:
            model, model_repr, model_class = load_model(
                        join(model_dirpath, "model.pt")
                    )
            model_ground_truth = load_ground_truth(model_dirpath)

            # synthetic_grad_norm = self.get_synthetic_gradients(model)
            input_grad = get_example_gradients(model, join(model_dirpath, 'clean-example-data/'))
            input_grad_norm = np.linalg.norm(input_grad, ord='fro')
            # input_grad_norm = np.linalg.norm(input_grad)
            # pdb.set_trace()
            # if model_class[-1].isnumeric():
            #     layer_id = int(model_class[-1])
            # else:
            #     layer_id = int(model_class[-2])
            # print("model_class is", model_class)
            # if model_class == 'MobileNetV2':
            #     layers = [n for n, _ in model.named_children()]
            #     print("layers[-1] is", layers[-1])
            #     weights = model_repr[layers[-1]+'.1.weight'].T
            # if model_class !='ResNet':
            #     print("model_class is", model_class)
            #     pdb.set_trace()
            # list(model.children())[-1].name
            layers = [n for n, _ in model.named_modules()]
            # print("layers[-1] is", layers[-1])
            weights = model_repr[layers[-1]+'.weight'].T
            # 
            # 
            # weights = model_repr[f'fc{layer_id}.weight'].T
            # biases = model_repr[f'fc{layer_id}.bias']
            s = np.linalg.svd(weights, compute_uv=False)[0:1]
            # model_class_ohe = enc.transform(np.array([model_class]).reshape(-1, 1)).toarray()[0]
            # concat the one hot encoded model class with the gradient score
            # X.append(np.concatenate((model_class_ohe, [input_grad_norm]), axis=0))
            X.append(np.concatenate(([input_grad_norm], s), axis=0))
            # X.append(np.concatenate(([input_grad_norm], [synthetic_grad_norm], s), axis=0))
            y.append(model_ground_truth)

        X = np.array(X)
        y = np.array(y)

        logging.info("Training BaggingClassifier with GradientBoostingClassifier base estimator...")
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=0)
        pipe = make_pipeline(StandardScaler(), gb)
        model = BaggingClassifier(base_estimator=pipe, n_estimators=500, random_state=0, n_jobs=-1)
        model.fit(X, y)
        logging.info("Training CalibratedClassifierCV...")
        calibrator = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
        calibrator.fit(X, y)
        
        model_filepath = join(output_parameters_dirpath, "model.bin")
        logging.info("Saving model...")
        with open(model_filepath, "wb") as fp:
            pickle.dump(calibrator, fp)

        # joblib.dump(enc, join(self.learned_parameters_dirpath, "ohe_encoder.bin"))

        # write_metaparameters()
        logging.info("Configuration done!")

# def configure(output_parameters_dirpath,
#               configure_models_dirpath,
#               parameter3):
    # logging.info('Using parameter3 = {}'.format(str(parameter3)))

    # logging.info('Configuring detector parameters with models from ' + configure_models_dirpath)

    # os.makedirs(output_parameters_dirpath, exist_ok=True)

    # logging.info('Writing configured parameter data to ' + output_parameters_dirpath)

    # arr = np.random.rand(100,100)
    # np.save(os.path.join(output_parameters_dirpath, 'numpy_array.npy'), arr)

    # with open(os.path.join(output_parameters_dirpath, "single_number.txt"), 'w') as fh:
    #     fh.write("{}".format(17))

    # example_dict = dict()
    # example_dict['keya'] = 2
    # example_dict['keyb'] = 3
    # example_dict['keyc'] = 5
    # example_dict['keyd'] = 7
    # example_dict['keye'] = 11
    # example_dict['keyf'] = 13
    # example_dict['keyg'] = 17

    # with open(os.path.join(output_parameters_dirpath, "dict.json"), mode='w', encoding='utf-8') as f:
    #     json.dump(example_dict, f, warnings=True, indent=2)


if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example')
    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--parameter1', type=int, help='An example tunable parameter.')
    parser.add_argument('--parameter2', type=float, help='An example tunable parameter.')
    parser.add_argument('--parameter3', type=str, help='An example tunable parameter.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    # logfile = '1006_all1.log'
    # logging.basicConfig(level=logging.INFO,
    #                     format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    #                     handlers=[
    #                     logging.FileHandler(logfile, "a"),
    #                     logging.StreamHandler()
    #                     ])
    # logging.info("example_trojan_detector.py launched")
    
    

    # Validate config file against schema
    config_json = None
    if args.metaparameters_filepath is not None:
        with open(args.metaparameters_filepath[0]()) as config_file:
            config_json = json.load(config_file)
            if args.parameter1 is None:
                args.parameter1 = config_json['parameters1']
            if args.parameter2 is None:
                args.parameter2 = config_json['parameters1']
    if args.schema_filepath is not None:
        with open(args.schema_filepath) as schema_file:
            schema_json = json.load(schema_file)

        # this throws a fairly descriptive error if validation fails
        jsonschema.validate(instance=config_json, schema=schema_json)

    # logging.info(args)

    if not args.configure_mode:
        if (args.model_filepath is not None and
            args.result_filepath is not None and
            args.scratch_dirpath is not None and
            args.examples_dirpath is not None and
            args.round_training_dataset_dirpath is not None and
            args.learned_parameters_dirpath is not None and
            args.parameter1 is not None and
            args.parameter2 is not None):

            # logging.info("Calling the trojan detector")
            example_trojan_detector(args.model_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath,
                                    args.parameter1, args.parameter2)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
            args.configure_models_dirpath is not None and
            args.parameter3 is not None):

            logging.info("Calling configuration mode")
            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      args.parameter3)
        else:
            logging.info("Required Configure-Mode parameters missing!")



