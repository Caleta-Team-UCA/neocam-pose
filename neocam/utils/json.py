import json


def read_json(path_json: str) -> dict:
    """Reads a JSON file as a dictionary"""
    with open(path_json) as f:
        data = json.load(f)
    return data
