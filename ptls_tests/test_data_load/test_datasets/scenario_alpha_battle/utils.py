import json

def save_json(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file)

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)
    
import pickle

def save_pkl(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_pkl(path):
    with open(path, 'rb') as file:
        return pickle.load(file)