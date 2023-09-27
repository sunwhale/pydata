# -*- coding: utf-8 -*-
"""

"""
import requests
import os


def load_server_status(url='http://127.0.0.1:5000/'):
    response = requests.get(url)
    result = response.json()
    specimens = result['data']
    specimens_status = {}
    for specimen in specimens:
        specimen_id = specimen['specimen_id']
        specimens_status[specimen_id] = {}
        for key in specimen.keys():
            specimens_status[specimen_id][key] = specimen[key]
    return specimens_status


def load_local_status(url='http://127.0.0.1:5000/'):
    response = requests.get(url)
    result = response.json()
    specimens = result['data']
    specimens_status = {}
    for specimen in specimens:
        specimen_id = specimen['specimen_id']
        specimens_status[specimen_id] = {}
        for key in specimen.keys():
            specimens_status[specimen_id][key] = specimen[key]
    return specimens_status


if __name__ == '__main__':
    data = {}
    specimens_status = load_status('https://www.sunjingyu.com/experiment/experiment_specimens_status/7')
    local_experiments_path = r'F:/GitHub/pydata/download/experiments'
    specimen_ids = [1]
    for specimen_id in specimen_ids:
        specimen_status = specimens_status[specimen_id]
        specimen_path = specimen_status['path']
        csv_file = os.path.join(specimen_path, 'timed.csv')

        try:
            data[specimen_id] = pd.read_csv(csv_file)
        except Exception as e:
            traceback.print_exc()
            print('error:' + npz_file)
