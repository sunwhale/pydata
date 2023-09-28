# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from numpy import ndarray
import requests
import os
import traceback
import pandas as pd
import matplotlib.pyplot as plt

from src.pydata.utils.dir_status import experiment_specimens_detail


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


def load_local_status(path, experiemnt_id):
    result = experiment_specimens_detail(path, experiemnt_id)
    specimens = result['data']
    specimens_status = {}
    for specimen in specimens:
        specimen_id = specimen['specimen_id']
        specimens_status[specimen_id] = {}
        for key in specimen.keys():
            specimens_status[specimen_id][key] = specimen[key]
    return specimens_status


def get_elastic_module(strain: ndarray, stress: ndarray) -> tuple[float, float]:
    # 使用最小二乘法进行一次多项式拟合（直线拟合）
    coefficients = np.polyfit(strain, stress, 1)
    E = float(coefficients[0])
    shift = float(coefficients[1])
    return E, shift


def get_elastic_limit(strain: ndarray, stress: ndarray, strain_start: float, strain_end: float,
                      threshold: float) -> tuple[list[float, float], float, float]:
    elastic_part = (strain > strain_start) & (strain < strain_end)
    elastic_strain = strain[elastic_part]
    elastic_stress = stress[elastic_part]
    E, shift = get_elastic_module(elastic_strain, elastic_stress)
    error = E * strain + shift - stress
    elastic_limit_index = strain[error < threshold].shape[0]
    elastic_limit = [float(strain[elastic_limit_index]), float(stress[elastic_limit_index])]
    return elastic_limit, E, shift


def get_fracture_strain(strain: ndarray, stress: ndarray, slop_criteria: float) -> tuple[float, float]:
    ultimate_stress_index = np.argmax(stress)
    stress_after_ultimate = stress[ultimate_stress_index:]
    strain_after_ultimate = strain[ultimate_stress_index:]
    diff_stress = np.diff(stress_after_ultimate)
    diff_strain = np.diff(strain_after_ultimate)
    none_zero_indices = (diff_strain != 0.0)
    derivative = diff_stress[none_zero_indices] / diff_strain[none_zero_indices]
    if not np.all(derivative < slop_criteria):
        fracture_strain_index = np.array(range(derivative.shape[0]))[derivative < slop_criteria][0]
        fracture_strain = strain_after_ultimate[1:][none_zero_indices][fracture_strain_index]
        fracture_stress = stress_after_ultimate[1:][none_zero_indices][fracture_strain_index]
        return fracture_strain, fracture_stress
    else:
        return strain[-1], stress[-1]


def get_peak_valley_time(strain: ndarray, stress: ndarray, time: ndarray, n: int):
    diff_stress = np.diff(stress)
    diff_strain = np.diff(strain)
    none_zero_indices = (diff_strain != 0.0)
    diff_strain = diff_strain[none_zero_indices]
    diff_strain_filter = np.convolve(diff_strain, np.ones((n,)) / n, mode='same')

    indices = np.where(np.diff(np.sign(diff_strain_filter)) != 0)[0]
    peak_valley_time = time[1:][none_zero_indices][1:][indices]

    plt.plot(time, strain)
    plt.plot(time[1:][none_zero_indices], np.sign(diff_strain_filter), ls='', marker='o')

    return peak_valley_time


def plot_elastic_limit(elastic_limit: list[float, float]) -> None:
    plt.plot(elastic_limit[0], elastic_limit[1], "bo", label="Elastic Breakpoint")
    plt.text(elastic_limit[0] + 0.01, elastic_limit[1] - 0.01, f"({elastic_limit[0]:.3f}, {elastic_limit[1]:.3f})",
             ha='left', va='bottom', fontsize=12)


def plot_elastic_module(elastic_limit: list[float, float], E: float, shift: float) -> None:
    x = np.linspace(0.0, 1.0, 100)
    y = E * x + shift
    plt.plot(x, y, ls='--', color='blue')
    plt.text(elastic_limit[0] + 0.02, elastic_limit[1] - 0.06, f"E={E:.3f}", ha='left', va='bottom', fontsize=12)


def plot_stress_strain(strain: ndarray, stress: ndarray) -> None:
    plt.plot(strain, stress, color='black', ls='-')


def plot_ultimate_stress(stress: ndarray) -> None:
    ultimate_stress_index = np.argmax(stress)
    ultimate_stress_strain = strain_exp[ultimate_stress_index]
    ultimate_stress = stress_exp[ultimate_stress_index]
    plt.plot(ultimate_stress_strain, ultimate_stress, 'ro', label="Max Stress Point")
    plt.text(ultimate_stress_strain + 0.04, ultimate_stress - 0.07, "({0}, {1})".format('εm', 'σm'), ha='right',
             va='top', fontsize=12)
    plt.axhline(y=max(stress_exp), color='r', linestyle='--')


def plot_fracture_strain(fracture_strain: float, fracture_stress: float) -> None:
    plt.axvline(x=fracture_strain, color='g', linestyle='--')
    plt.text(fracture_strain, fracture_stress - 0.12, f"({fracture_strain:.3f}, {fracture_stress:.3f})", ha='right',
             va='bottom', fontsize=12)
    plt.plot(fracture_strain, fracture_stress, "go", label="Endpoint")


if __name__ == '__main__':
    data = {}
    local_experiments_path = r'F:/GitHub/pydata/download/experiments'
    specimens_status = load_local_status(local_experiments_path, 8)

    specimen_ids = [1]

    for specimen_id in specimen_ids:
        specimen_status = specimens_status[specimen_id]
        specimen_path = specimen_status['path']
        csv_file = os.path.join(specimen_path, 'timed.csv')
        try:
            data[specimen_id] = pd.read_csv(csv_file)
        except Exception as e:
            traceback.print_exc()
            print('error:' + csv_file)

        time = np.array(data[specimen_id]['Time_s'])
        strain_exp = np.array(data[specimen_id]['Strain'])
        stress_exp = np.array(data[specimen_id]['Stress_MPa'])

        elastic_limit, E, shift = get_elastic_limit(strain_exp, stress_exp, 0.005, 0.1, 0.01)
        # fracture_strain, fracture_stress = get_fracture_strain(strain_exp, stress_exp, -50.0)
        get_peak_valley_time(strain_exp, stress_exp, time, 10)

        # plot_stress_strain(strain_exp, stress_exp)
        # plot_elastic_limit(elastic_limit)
        # plot_elastic_module(elastic_limit, E, shift)
        # plot_ultimate_stress(stress_exp)
        # plot_fracture_strain(fracture_strain, fracture_stress)
        plt.show()
