import os

from neocam.pose.analysis import Analysis
from neocam.utils.json import read_json

from unittest import TestCase


def test_analysis_to_json():
    path_json = "test.json"
    analysis = Analysis(size=240, frequency=1, plot_series=False, dummy=False)
    analysis.to_json(path_json)
    # Read the json we just stored
    dict_json = read_json(path_json)
    os.remove(path_json)
    # Assert dictionaries are equal
    case = TestCase()
    case.assertDictEqual(dict_json, analysis.dict)


if __name__ == "__main__":
    test_analysis_to_json()
