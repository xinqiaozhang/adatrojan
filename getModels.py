import logging
import os
import json
import jsonpickle
import pickle
import numpy as np
from collections import OrderedDict

import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper

import csv

def csv_to_list_of_lists(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data



def pad_arrays_to_length(arr_list):
    target_length = 0
    padded_arrays = []

    # Iterate through each array in the list
    for folder, arr in arr_list:
        target_length = max(target_length, len(arr))

    for folder, arr in arr_list:
        # Calculate the number of zeros needed for padding
        num_zeros_to_pad = max(target_length - len(arr), 0)

        # Pad the array with zeros using NumPy's pad function
        padded_arr = np.pad(arr, (0, num_zeros_to_pad), mode='constant')

        # Append the padded array to the list
        padded_arrays.append([folder, padded_arr])

    return padded_arrays


def getModels(directory_path, csv_file_path):
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory path.")
        return

    # Get a list of all folders in the given directory
    folders_list = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]

    weightList = []

    for folder in folders_list:
        _, modelRep, _ = load_model(directory_path + "/" + folder + "/model.pt")
        weights = []
        for key, value in list(modelRep.items()):
            # Convert NumPy array to PyTorch tensor and then flatten it
            tensor_value = torch.tensor(value)
            flattened_tensor = torch.flatten(tensor_value).tolist()
            weights.extend(flattened_tensor)  # Extend the list with the flattened tensor values
        weightList.append([int(folder[3:]), weights])
        
    data = csv_to_list_of_lists(csv_file_path)

    csvData = []
    for element in data:
        if 'id-' in element[0]:
            element[0] = int(element[0][3:])
            csvData.append(element)

    csvData = sorted(csvData, key=lambda x: x[0])

    weightList = pad_arrays_to_length(weightList)
    weightList = sorted(weightList, key=lambda x: x[0])
    
    for element in weightList:
        #element.append(csvData[element[0]][5])
        #element.append(csvData[element[0]][6])
        if csvData[element[0]][4] == "clean":
            element.append(0)
        elif csvData[element[0]][4] == "triggered":
            element.append(1)
        else:
            print("ERROR")
        values_to_append = np.array([0, 0])
        if csvData[element[0]][1] == "SimplifiedRLStarter":
            values_to_append[0] = 1
        elif csvData[element[0]][1] == "BasicFCModel":
            values_to_append[1] = 1
        else:
            print("\n\nERROR: ", csvData[element[0]][1])
        element[1] = np.concatenate((element[1], values_to_append), axis = 0)
        #print("ID: ", element[0], " VAL: ", csvData[element[0]])
    weightList = [x[1:] for x in weightList]
    return weightList

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)

input_size = 27446
hidden_size = 300
hidden_size2 = 150
model = BinaryClassifier(input_size, hidden_size, hidden_size2)
model = model.to(torch.float32)

def run(data, test_size = 0.2):
    #random.shuffle(data)
    features_list, labels_list = zip(*data)

    features_tensor = torch.tensor(features_list, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.float32)

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_tensor)

    train_features, test_features, train_labels, test_labels = train_test_split(
        normalized_features, labels_tensor, test_size=test_size, random_state=42
    )

    class CustomDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, index):
            feature = torch.tensor(self.features[index], dtype=torch.float32)
            label = self.labels[index]
            return feature, label

    batch_size = 32
    
    train_dataset = CustomDataset(train_features, train_labels)
    test_dataset = CustomDataset(test_features, test_labels)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 200

    best_accuracy = 0.0
    best_checkpoint = -1
    checkpoint_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data_loader)
        if (epoch + 1) == 1 or ((epoch + 1)%10 == 0):
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()

    accuracy = 100 * correct / total
    checkpoint_accuracies.append(accuracy)
    print(f"NN Test Accuracy: {accuracy:.2f}%")
    bestPath = ""

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_checkpoint = epoch + 1
        torch.save(model.state_dict(), f"binary_classifier_dict_{best_checkpoint}.pt")
        #torch.save(model, f"binary_classifier_checkpoint_{best_checkpoint}.pt")
        #bestPath = f"binary_classifier_checkpoint_{best_checkpoint}.pt"

    #print("Best checkpoint:", best_checkpoint)

    model1, dict1, str1 = load_model(bestPath)

    RFModel = RandomForestClassifier(n_estimators=100, random_state=100, max_depth=100)

    RFModel.fit(train_features, train_labels)

    accuracy = RFModel.score(test_features, test_labels)*100.0


    print(f"RF Test Accuracy: {accuracy:.2f}%")


def runTraining(directory_path, csv_file_path):
    data = getModels(directory_path, csv_file_path)
    print("TEST SIZE: ", 10, "%")
    run(data, test_size=0.1)
    # for test_size in range(5, 30, 5):
    #     print("TEST SIZE: ", test_size, "%")
    #     run(data, test_size=test_size/100.0)

def load_model(model_filepath: str) -> (dict, str):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to model.pt file

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """
    model = torch.load(model_filepath)
    model_class = model.__class__.__name__
    model_repr = OrderedDict(
        {layer: tensor.numpy() for (layer, tensor) in model.state_dict().items()}
    )

    return model, model_repr, model_class

def main():
    runTraining('../lw-train/models/', '../lw-train/METADATA.csv')

if __name__ == "__main__":
    main()