# -*- coding: utf-8 -*-
"""

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import array, ndarray
from scipy import interpolate
from scipy.optimize import fmin

from src.pydata.utils.mechanics import load_local_status, get_elastic_limit


def finite_element_solution(paras: list, constants: list, strain: ndarray, time: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    import sys
    PYFEM_PATH = r'F:\Github\pyfem\src'
    sys.path.insert(0, PYFEM_PATH)
    from pyfem.Job import Job
    from pyfem.io.BaseIO import BaseIO

    job = Job(r'F:\Github\pyfem\examples\mechanical\1element\hex8_visco\Job-1.toml')

    total_time = time[-1]
    amplitude = [[time[i], strain[i]] for i in range(len(time))]

    BaseIO.is_read_only = False
    nu = 0.14
    material_data = []
    material_data.append(constants[0])
    material_data.append(nu)
    if len(constants) > 1:
        m = int(len(paras))
        for i in range(m):
            material_data.append(paras[i])
            material_data.append(constants[i + 1])
    else:
        m = int(len(paras) / 2)
        for i in range(m):
            material_data.append(paras[2 * i])
            material_data.append(constants[2 * i + 1])
    job.props.materials[0].data = material_data
    job.props.solver.total_time = total_time
    job.props.solver.max_dtime = total_time / 50.0
    job.props.solver.initial_dtime = total_time / 50.0
    job.props.amplitudes[0].data = amplitude
    job.assembly.__init__(job.props)

    _, e11, s11, t = job.run()

    return e11, s11, t


def finite_element_solution_damage(paras: list, constants: list, strain: ndarray, time: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    import sys
    PYFEM_PATH = r'F:\Github\pyfem\src'
    sys.path.insert(0, PYFEM_PATH)
    from pyfem.Job import Job
    from pyfem.io.BaseIO import BaseIO

    job = Job(r'F:\Github\pyfem\examples\mechanical_phase\1element\hex8_visco\Job-1.toml')

    total_time = time[-1]
    amplitude = [[time[i], strain[i]] for i in range(len(time))]

    BaseIO.is_read_only = False
    material_data = []
    material_data.append(constants[0])
    material_data.append(constants[1])
    m = int(len(constants) / 2) - 1
    for i in range(m):
        material_data.append(constants[2 * (i + 1)])
        material_data.append(constants[2 * (i + 1) + 1])
    job.props.materials[0].data = material_data
    job.props.materials[1].data = paras
    job.props.solver.total_time = total_time
    job.props.solver.max_dtime = total_time / 50.0
    job.props.solver.initial_dtime = total_time / 50.0
    job.props.amplitudes[0].data = amplitude
    job.assembly.__init__(job.props)

    _, e11, s11, t = job.run()

    return e11, s11, t


def analytical_tensile_solution(paras: list, constants: list, strain: ndarray, time: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    t = time
    dt = np.diff(t)
    strain_rate = np.diff(strain) / np.diff(t)
    dt = np.concatenate((dt, dt[-1:]))
    strain_rate = np.concatenate((strain_rate, strain_rate[-1:]))

    E_inf = constants[0]
    h = []
    if len(constants) > 1:
        m = int(len(paras))
        for i in range(m):
            h.append([0.])
            E_i = paras[i]
            tau_i = constants[i + 1]
            for j in range(1, len(dt)):
                h[i].append( np.exp(-dt[j] / tau_i) * h[i][j - 1] + tau_i * E_i * (1 - np.exp(-dt[j] / tau_i)) * strain_rate[j])
    else:
        m = int(len(paras) / 2)
        for i in range(m):
            h.append([0.])
            E_i = paras[2 * i]
            tau_i = paras[2 * i + 1]
            for j in range(1, len(dt)):
                h[i].append( np.exp(-dt[j] / tau_i) * h[i][j - 1] + tau_i * E_i * (1 - np.exp(-dt[j] / tau_i)) * strain_rate[j])

    ha = np.array(h).transpose()
    stress = E_inf * strain + ha.sum(axis=1)
    return strain, stress, time


def analytical_relax_solution(paras: list, constants: list, strain: ndarray, time: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    E_inf = constants[0]
    stress = E_inf * strain
    if len(constants) > 1:
        m = int(len(paras))
        for i in range(m):
            E_i = paras[i]
            tau_i = constants[i + 1]
            stress += E_i * np.exp(-time / tau_i) * strain
    else:
        m = int(len(paras) / 2)
        for i in range(m):
            E_i = paras[2 * i]
            tau_i = paras[2 * i + 1]
            stress += E_i * np.exp(-time / tau_i) * strain
    return strain, stress, time


def get_experiment_data(experiments_path: str, experiment_id: int, specimen_ids: list) -> tuple[dict, dict]:
    experiment_data = {}
    experiment_status = load_local_status(experiments_path, experiment_id)
    for specimen_id in specimen_ids:
        specimen_status = experiment_status[specimen_id]
        specimen_path = specimen_status['path']
        csv_file = os.path.join(specimen_path, 'timed.csv')
        try:
            experiment_data[specimen_id] = pd.read_csv(csv_file)
        except Exception as e:
            print('error:' + csv_file)
    return experiment_data, experiment_status


def filter_time_stress_strain(data: dict) -> dict:
    """
    对数据进行预处理，去除相同时间的数据点
    """
    processed_data = {}
    for key in data.keys():
        time = array(data[key]['Time_s'])
        strain = array(data[key]['Strain'])
        stress = array(data[key]['Stress_MPa'])

        # 转换为柯西应力和对数应变
        strain = np.log(strain + 1.0)
        stress = stress * (1.0 + stress)

        is_duplicate = np.full(len(time), False)
        values, counts = np.unique(time, return_index=True)
        is_duplicate[counts] = True
        time = time[is_duplicate]
        strain = strain[is_duplicate]
        stress = stress[is_duplicate]
        f_strain_stress = interpolate.interp1d(strain, stress, kind='linear', fill_value='extrapolate')
        f_time_strain = interpolate.interp1d(time, strain, kind='linear', fill_value='extrapolate')
        f_time_stress = interpolate.interp1d(time, stress, kind='linear', fill_value='extrapolate')
        processed_data[key] = {'Time_s': time,
                               'Strain': strain,
                               'Stress_MPa': stress,
                               'f_strain_stress': f_strain_stress,
                               'f_time_strain': f_time_strain,
                               'f_time_stress': f_time_stress}
    return processed_data


def partial_by_elastic_limit(data: dict) -> dict:
    """
    对数据进行预处理，截取弹性部分
    """
    processed_data = {}
    for key in data.keys():
        time = array(data[key]['Time_s'])
        strain = array(data[key]['Strain'])
        stress = array(data[key]['Stress_MPa'])
        # try:
        #     elastic_limit, E, shift = get_elastic_limit(strain, stress, strain_start=0.005, strain_end=0.1,
        #                                                 threshold=10)
        #     elastic_indices = strain < elastic_limit[0]
        #     time = time[elastic_indices]
        #     strain = strain[elastic_indices]
        #     stress = stress[elastic_indices]
        # except:
        #     pass

        elastic_indices = strain < 0.5
        time = time[elastic_indices]
        strain = strain[elastic_indices]
        stress = stress[elastic_indices]

        target_rows = 100

        if len(time) > target_rows:
            # 缩减到1000行，不超过1000行则不会缩减
            total_rows = len(time)
            interval = total_rows // target_rows
            time = time[::interval]
            strain = strain[::interval]
            stress = stress[::interval]

        f_strain_stress = interpolate.interp1d(strain, stress, kind='linear', fill_value='extrapolate')
        f_time_strain = interpolate.interp1d(time, strain, kind='linear', fill_value='extrapolate')
        f_time_stress = interpolate.interp1d(time, stress, kind='linear', fill_value='extrapolate')
        processed_data[key] = {'Time_s': time,
                               'Strain': strain,
                               'Stress_MPa': stress,
                               'f_strain_stress': f_strain_stress,
                               'f_time_strain': f_time_strain,
                               'f_time_stress': f_time_stress}
    return processed_data


def cal_tensile_cost(data: dict, paras: list, constants: list):
    cost = 0.0
    for key in data.keys():
        time = data[key]['Time_s']
        strain_exp = data[key]['Strain']
        stress_exp = data[key]['Stress_MPa']
        f_time_stress = data[key]['f_time_stress']
        f_time_strain = data[key]['f_time_strain']
        stress_exp_max = max(stress_exp)
        # strain_sim, stress_sim, time_sim = analytical_tensile_solution(paras, constants, strain_exp, time)
        strain_sim, stress_sim, time_sim = finite_element_solution_damage(paras, constants, strain_exp, time)
        stress_exp = f_time_stress(time_sim)
        cost += np.sum(((stress_exp - stress_sim) / stress_exp_max) ** 2, axis=0) / len(time)
    return cost


def cal_relax_cost(data: dict, paras: list, constants: list):
    cost = 0.0
    for key in data.keys():
        time = data[key]['Time_s']
        f_time_stress = data[key]['f_time_stress']
        f_time_strain = data[key]['f_time_strain']
        time = np.linspace(time[0], time[-1], 100)
        strain_exp = f_time_strain(time)
        stress_exp = f_time_stress(time)
        stress_exp_max = max(stress_exp)
        strain_sim, stress_sim, time_sim = analytical_relax_solution(paras, constants, strain_exp, time)
        cost += np.sum(((stress_exp - stress_sim) / stress_exp_max) ** 2, axis=0) / len(time)
    return cost


def func(x: list):
    cost = 0.0
    cost += cal_tensile_cost(processed_tensile_data, x, constants)
    # cost += cal_relax_cost(processed_relax_data, x, constants)
    punish = 0.0
    y = cost
    for i in range(len(x)):
        punish += (max(0, -x[i])) ** 2
    y += 1e16 * punish
    print(y)
    return y


if __name__ == '__main__':
    local_experiments_path = r'F:/GitHub/pydata/download/experiments'
    experiment_id = 7
    # tensile_specimen_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    tensile_specimen_ids = [1, 4, 7]
    # tensile_specimen_ids = [11, 14, 17]
    # tensile_specimen_ids = [18, 23, 24]
    tensile_experiment_data, tensile_experiment_status = get_experiment_data(local_experiments_path, experiment_id,
                                                                             tensile_specimen_ids)
    processed_tensile_data = partial_by_elastic_limit(filter_time_stress_strain(tensile_experiment_data))

    experiment_id = 9
    relax_specimen_ids = [1]
    # relax_specimen_ids = [4]
    # relax_specimen_ids = [9]
    relax_experiment_data, relax_experiment_status = get_experiment_data(local_experiments_path, experiment_id,
                                                                         relax_specimen_ids)
    processed_relax_data = partial_by_elastic_limit(filter_time_stress_strain(relax_experiment_data))

    # paras_0 = [1.0, 2e-1, 1.0, 5.0e1, 1.0, 2.0e3]
    # constants = [0.97]

    paras_0 = [0.01656102, 0.00236847]
    # constants = [0.95, 0.5, 40.0, 1500.0]
    # constants = [0.95, 0.1, 2.0, 1000.0]
    constants = [0.95, 0.14, 7.49243691, 0.1, 5.92043313, 2.0, 1.37908231, 1000.0]
    paras = fmin(func, paras_0, maxiter=20, ftol=1e-4, xtol=1e-4, disp=True)
    print(paras)

    # for tau_1 in np.linspace(4e-1, 8e-1, 5):
    #     for tau_2 in np.linspace(1e1, 1e2, 5):
    #         for tau_3 in np.linspace(1e3, 2e3, 5):
    #             constants = [0.95, tau_1, tau_2, tau_3]
    #             paras = fmin(func, paras_0, maxiter=2000, ftol=1e-4, xtol=1e-4, disp=False)  # 优化后的参数
    #             print(tau_1, tau_2, tau_3, func(paras))

    for specimen_id in tensile_specimen_ids:
        time_exp = processed_tensile_data[specimen_id]['Time_s']
        stress_exp = processed_tensile_data[specimen_id]['Stress_MPa']
        strain_exp = processed_tensile_data[specimen_id]['Strain']
        f_strain_stress = processed_tensile_data[specimen_id]['f_strain_stress']
        plt.plot(strain_exp, stress_exp, marker='o')
        # strain_sim, stress_sim, time_sim = analytical_tensile_solution(paras, constants, strain_exp, time_exp)
        strain_sim, stress_sim, time_sim = finite_element_solution_damage(paras, constants, strain_exp, time_exp)
        plt.plot(strain_sim, stress_sim, color='red')
    plt.show()

    for specimen_id in relax_specimen_ids:
        time_exp = processed_relax_data[specimen_id]['Time_s']
        stress_exp = processed_relax_data[specimen_id]['Stress_MPa']
        strain_exp = processed_relax_data[specimen_id]['Strain']
        f_strain_stress = processed_relax_data[specimen_id]['f_strain_stress']
        plt.plot(time_exp, stress_exp, marker='o')
        # strain_sim, stress_sim, time_sim = analytical_relax_solution(paras, constants, strain_exp, time_exp)
        strain_exp[0] = 0.0
        strain_sim, stress_sim, time_sim = finite_element_solution(paras, constants, strain_exp, time_exp)
        plt.plot(time_sim, stress_sim, color='red')
    plt.show()