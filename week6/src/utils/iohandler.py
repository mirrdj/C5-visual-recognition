import json
import csv
import os

def read_json(dir, file):
    file = os.path.join(dir, file)

    with open(file, 'r') as file:
        data = json.load(file)

    return data


def read_csv(dir, file):
    file = os.path.join(dir, file)
    entries = []

    with open(file, 'r') as file:
        for line in file:
            entry = line.split(",")
            entries.append(entry)
        
    return entries

