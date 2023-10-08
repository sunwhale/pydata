# -*- coding: utf-8 -*-
"""

"""
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from numpy import ndarray

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
    error = abs(E * strain + shift - stress)
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


def get_peak_valley_time(strain: ndarray, stress: ndarray, time: ndarray, n: int) -> tuple[ndarray, ndarray]:
    diff_stress = np.diff(stress)
    diff_strain = np.diff(strain)
    none_zero_indices = (diff_strain != 0.0)
    diff_strain = diff_strain[none_zero_indices]

    for i in range(n, n + 20):
        diff_strain_filter = np.convolve(diff_strain, np.ones((i,)) / i, mode='same')
        if np.sum(np.sign(diff_strain_filter) == 0) == 0:
            break

    peak_indices = np.where(np.diff(np.sign(diff_strain_filter)) < 0)[0]
    valley_indices = np.where(np.diff(np.sign(diff_strain_filter)) > 0)[0]

    peak_time = time[1:][none_zero_indices][1:][peak_indices]
    valley_time = time[1:][none_zero_indices][1:][valley_indices]

    return peak_time, valley_time


def get_loading_time(strain: ndarray, stress: ndarray, time: ndarray, peak_time: ndarray, valley_time: ndarray) -> dict:
    data = {}
    is_in_time_range = (time < peak_time[0])
    data[0] = {}
    data[0]['Time_s'] = time[is_in_time_range]
    data[0]['Stress_MPa'] = stress[is_in_time_range]
    data[0]['Strain'] = strain[is_in_time_range]

    for i in range(len(valley_time) - 1):
        is_in_time_range = (time > valley_time[i]) & (time < peak_time[i + 1])
        data[i + 1] = {}
        data[i + 1]['Time_s'] = time[is_in_time_range]
        data[i + 1]['Stress_MPa'] = stress[is_in_time_range]
        data[i + 1]['Strain'] = strain[is_in_time_range]
    return data


def partial_by_stress(data: dict, stress_limit: float) -> dict:
    processed_data = {}
    for key in data.keys():
        indices = data[key]['Stress_MPa'] > stress_limit
        processed_data[key] = {}
        processed_data[key]['Stress_MPa'] = data[key]['Stress_MPa'][indices]
        processed_data[key]['Strain'] = data[key]['Strain'][indices]
        processed_data[key]['Time_s'] = data[key]['Time_s'][indices]
    return processed_data


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

    specimen_ids = [1, 2, 6, 7, 11, 12]
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
        peak_time, valley_time = get_peak_valley_time(strain_exp, stress_exp, time, 7)
        specimen_data = get_loading_time(strain_exp, stress_exp, time, peak_time, valley_time)
        processed_specimen_data = partial_by_stress(specimen_data, stress_limit=0.004)

        # plot_stress_strain(strain_exp, stress_exp)

        for key, value in processed_specimen_data.items():
            if key == 0:
                elastic_limit, E, shift = get_elastic_limit(value['Strain'], value['Stress_MPa'], value['Strain'][0] + 0.008, value['Strain'][0] + 0.01, 0.002)
                value['elastic_limit'] = elastic_limit
                value['E'] = E
                value['shift'] = shift
            else:
                elastic_limit, E, shift = get_elastic_limit(value['Strain'], value['Stress_MPa'], value['Strain'][0], value['Strain'][0] + 0.006, 0.004)
                value['elastic_limit'] = elastic_limit
                value['E'] = E
                value['shift'] = shift

        strain = [value['Strain'][0] for _, value in processed_specimen_data.items()]
        E_init = processed_specimen_data[0]['E']
        damage = [1.0 - value['E']/E_init for _, value in processed_specimen_data.items()]

        plt.plot(strain, damage, marker='o', ls='', color='red')

            # plot_elastic_limit(elastic_limit)
            # plot_elastic_module(elastic_limit, E, shift)
        # plot_ultimate_stress(stress_exp)
        # plot_fracture_strain(fracture_strain, fracture_stress)

        specimen_ids = [3, 4, 8, 9, 13, 14]
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
            peak_time, valley_time = get_peak_valley_time(strain_exp, stress_exp, time, 7)
            specimen_data = get_loading_time(strain_exp, stress_exp, time, peak_time, valley_time)
            processed_specimen_data = partial_by_stress(specimen_data, stress_limit=0.004)

            # plot_stress_strain(strain_exp, stress_exp)

            for key, value in processed_specimen_data.items():
                if key == 0:
                    elastic_limit, E, shift = get_elastic_limit(value['Strain'], value['Stress_MPa'], value['Strain'][0] + 0.005, value['Strain'][0] + 0.01,
                                                                0.002)
                    value['elastic_limit'] = elastic_limit
                    value['E'] = E
                    value['shift'] = shift
                else:
                    elastic_limit, E, shift = get_elastic_limit(value['Strain'], value['Stress_MPa'], value['Strain'][0], value['Strain'][0] + 0.006, 0.004)
                    value['elastic_limit'] = elastic_limit
                    value['E'] = E
                    value['shift'] = shift

            strain = [value['Strain'][0] for _, value in processed_specimen_data.items()]
            E_init = processed_specimen_data[0]['E']
            damage = [1.0 - value['E'] / E_init for _, value in processed_specimen_data.items()]

            plt.plot(strain, damage, marker='o', ls='', color='blue')

    plt.show()
