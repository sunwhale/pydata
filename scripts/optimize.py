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

from src.pydata.utils.mechanics import load_local_status, get_elastic_limit, get_fracture_strain


def finite_element_solution(paras: list, constants: dict, strain: ndarray, time: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    计算耦合相场断裂的广义Maxwell模型的有限元解
    """
    import sys
    PYFEM_PATH = r'F:\Github\pyfem\src'
    sys.path.insert(0, PYFEM_PATH)
    from pyfem.Job import Job
    from pyfem.io.BaseIO import BaseIO

    job = Job(r'F:\Github\pyfem\examples\mechanical\1element\hex8_visco\Job-1.toml')

    total_time = time[-1]
    amplitude = [[time[i], strain[i]] for i in range(len(time))]

    BaseIO.is_read_only = False

    material_data = []
    material_data.append(constants['E_inf'])
    material_data.append(constants['nu'])
    tau_number = constants['tau_number']
    tau = constants['tau']

    for i in range(tau_number):
        if len(tau) > 0:
            material_data.append(paras[i])
            material_data.append(tau[i])
        else:
            material_data.append(paras[2 * i])
            material_data.append(paras[2 * i + 1])

    job.props.materials[0].data = material_data
    job.props.solver.total_time = total_time
    job.props.solver.max_dtime = total_time / 50.0
    job.props.solver.initial_dtime = total_time / 50.0
    job.props.amplitudes[0].data = amplitude
    job.assembly.__init__(job.props)

    _, e11, s11, t = job.run()

    return e11, s11, t


def finite_element_solution_damage(paras: list, constants: dict, strain: ndarray, time: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    计算耦合相场断裂的广义Maxwell模型的有限元解
    """
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
    material_data.append(constants['E_inf'])
    material_data.append(constants['nu'])
    tau_number = constants['tau_number']
    tau = constants['tau']
    E = constants['E']
    for i in range(tau_number):
        material_data.append(tau[i])
        material_data.append(E[i])

    job.props.materials[0].data = material_data
    job.props.materials[1].data = paras
    job.props.solver.total_time = total_time
    job.props.solver.max_dtime = total_time / 50.0
    job.props.solver.initial_dtime = total_time / 50.0
    job.props.amplitudes[0].data = amplitude
    job.assembly.__init__(job.props)

    _, e11, s11, t = job.run()

    return e11, s11, t


def analytical_tensile_solution(paras: list, constants: dict, strain: ndarray, time: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    计算单调拉伸实验广义Maxwell模型的解析解
    """
    t = time
    dt = np.diff(t)
    strain_rate = np.diff(strain) / np.diff(t)
    dt = np.concatenate((dt, dt[-1:]))
    strain_rate = np.concatenate((strain_rate, strain_rate[-1:]))

    E_inf = constants['E_inf']
    tau_number = constants['tau_number']
    tau = constants['tau']

    h = []

    for i in range(tau_number):
        h.append([0.])
        if len(tau) > 0:
            E_i = paras[i]
            tau_i = tau[i]
        else:
            E_i = paras[2 * i]
            tau_i = paras[2 * i + 1]
        for j in range(1, len(dt)):
            h[i].append(np.exp(-dt[j] / tau_i) * h[i][j - 1] + tau_i * E_i * (1 - np.exp(-dt[j] / tau_i)) * strain_rate[j])

    ha = np.array(h).transpose()
    stress = E_inf * strain + ha.sum(axis=1)
    return strain, stress, time


def analytical_relax_solution(paras: list, constants: dict, strain: ndarray, time: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    计算应力松弛实验广义Maxwell模型的解析解
    """
    E_inf = constants['E_inf']
    tau_number = constants['tau_number']
    tau = constants['tau']

    stress = E_inf * strain

    for i in range(tau_number):
        if len(tau) > 0:
            E_i = paras[i]
            tau_i = tau[i]
        else:
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


def create_data_dict(time: ndarray, strain: ndarray, stress: ndarray) -> dict:
    f_strain_stress = interpolate.interp1d(strain, stress, kind='linear', fill_value='extrapolate')
    f_time_strain = interpolate.interp1d(time, strain, kind='linear', fill_value='extrapolate')
    f_time_stress = interpolate.interp1d(time, stress, kind='linear', fill_value='extrapolate')
    data = {'Time_s': time,
            'Strain': strain,
            'Stress_MPa': stress,
            'f_strain_stress': f_strain_stress,
            'f_time_strain': f_time_strain,
            'f_time_stress': f_time_stress}
    return data


def preproc_data(data: dict, strain_shift: float) -> dict:
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

        # 去除时间重复的数据点
        is_duplicate = np.full(len(time), False)
        values, counts = np.unique(time, return_index=True)
        is_duplicate[counts] = True
        time = time[is_duplicate]
        strain = strain[is_duplicate]
        stress = stress[is_duplicate]

        # 应变平移
        shift_indices = strain < strain_shift
        shift = len(time[shift_indices])
        strain = strain[shift:] - strain_shift
        time = time[shift:] - time[shift]
        stress = stress[shift:]

        processed_data[key] = create_data_dict(time, strain, stress)

    return processed_data


def partial_by_elastic_limit(data: dict, strain_start: float = 0.005, strain_end: float = 0.1, threshold: float = 0.1) -> dict:
    """
    对数据进行预处理，截取弹性极限之前的部分
    """
    processed_data = {}
    for key in data.keys():
        time = array(data[key]['Time_s'])
        strain = array(data[key]['Strain'])
        stress = array(data[key]['Stress_MPa'])

        try:
            elastic_limit, E, shift = get_elastic_limit(strain, stress, strain_start, strain_end, threshold)
            elastic_indices = strain < elastic_limit[0]
            time = time[elastic_indices]
            strain = strain[elastic_indices]
            stress = stress[elastic_indices]
        except:
            pass

        processed_data[key] = create_data_dict(time, strain, stress)

    return processed_data


def partial_by_fracture_strain(data: dict) -> dict:
    """
    对数据进行预处理，截取断裂之前的部分
    """
    processed_data = {}
    for key in data.keys():
        time = array(data[key]['Time_s'])
        strain = array(data[key]['Strain'])
        stress = array(data[key]['Stress_MPa'])

        try:
            fracture_strain, fracture_stress = get_fracture_strain(strain, stress, -50.0)
            elastic_indices = strain < fracture_strain
            time = time[elastic_indices]
            strain = strain[elastic_indices]
            stress = stress[elastic_indices]
        except:
            pass

        processed_data[key] = create_data_dict(time, strain, stress)

    return processed_data


def reduce_to_target_rows(data: dict, target_rows: int = 100) -> dict:
    """
    通过等间距取数，减少数据到制定行数
    """
    processed_data = {}
    for key in data.keys():
        time = array(data[key]['Time_s'])
        strain = array(data[key]['Strain'])
        stress = array(data[key]['Stress_MPa'])

        if len(time) > target_rows:
            total_rows = len(time)
            interval = total_rows // target_rows
            time = time[::interval]
            strain = strain[::interval]
            stress = stress[::interval]

        processed_data[key] = create_data_dict(time, strain, stress)

    return processed_data


def cal_tensile_cost(data: dict, paras: list, constants: dict):
    """
    计算单调拉伸试验的计算值与预测值的误差
    """
    cost = 0.0
    for key in data.keys():
        time_exp = data[key]['Time_s']
        strain_exp = data[key]['Strain']
        stress_exp = data[key]['Stress_MPa']
        if constants['mode'] == 'analytical':
            strain_sim, stress_sim, time_sim = analytical_tensile_solution(paras, constants, strain_exp, time_exp)
        elif constants['mode'] == 'fem':
            strain_sim, stress_sim, time_sim = finite_element_solution_damage(paras, constants, strain_exp, time_exp)
            f_time_stress = data[key]['f_time_stress']
            stress_exp = f_time_stress(time_sim)
        else:
            raise NotImplementedError
        cost += np.sum(((stress_exp - stress_sim) / max(stress_exp)) ** 2, axis=0) / len(time_sim)
    return cost


def cal_relax_cost(data: dict, paras: list, constants: dict):
    """
    计算应力松弛试验的计算值与预测值的误差
    """
    cost = 0.0
    for key in data.keys():
        time_exp = data[key]['Time_s']
        f_time_stress = data[key]['f_time_stress']
        f_time_strain = data[key]['f_time_strain']

        # 在时间轴均匀分布应力松弛实验得到的数据
        time_exp = np.linspace(time_exp[0], time_exp[-1], 100)
        strain_exp = f_time_strain(time_exp)
        stress_exp = f_time_stress(time_exp)

        if constants['mode'] == 'analytical':
            strain_sim, stress_sim, time_sim = analytical_relax_solution(paras, constants, strain_exp, time_exp)
        elif constants['mode'] == 'fem':
            stress_exp[0] = 0.0  # 在有限元计算应力松弛时，赋予初始应变值为0
            strain_sim, stress_sim, time_sim = finite_element_solution_damage(paras, constants, strain_exp, time_exp)
            stress_exp = f_time_stress(time_sim)
        else:
            raise NotImplementedError
        cost += np.sum(((stress_exp - stress_sim) / max(stress_exp)) ** 2, axis=0) / len(time_exp)
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


def plot_tensile(specimen_ids: list, data: dict, paras: list, constants: dict) -> None:
    for specimen_id in specimen_ids:
        time_exp = data[specimen_id]['Time_s']
        stress_exp = data[specimen_id]['Stress_MPa']
        strain_exp = data[specimen_id]['Strain']
        plt.plot(strain_exp, stress_exp, marker='o')

        if constants['mode'] == 'analytical':
            strain_sim, stress_sim, time_sim = analytical_tensile_solution(paras, constants, strain_exp, time_exp)
        elif constants['mode'] == 'fem':
            strain_sim, stress_sim, time_sim = finite_element_solution_damage(paras, constants, strain_exp, time_exp)
        else:
            raise NotImplementedError
        plt.plot(strain_sim, stress_sim, color='red')
    plt.show()


def plot_relax(specimen_ids: list, data: dict, paras: list, constants: dict) -> None:
    for specimen_id in specimen_ids:
        time_exp = data[specimen_id]['Time_s']
        stress_exp = data[specimen_id]['Stress_MPa']
        strain_exp = data[specimen_id]['Strain']
        plt.plot(time_exp, stress_exp, marker='o')

        if constants['mode'] == 'analytical':
            strain_sim, stress_sim, time_sim = analytical_relax_solution(paras, constants, strain_exp, time_exp)
        elif constants['mode'] == 'fem':
            strain_exp[0] = 0.0
            strain_sim, stress_sim, time_sim = finite_element_solution_damage(paras, constants, strain_exp, time_exp)
        else:
            raise NotImplementedError
        plt.plot(time_sim, stress_sim, color='red')
    plt.show()


if __name__ == '__main__':
    local_experiments_path = r'F:/GitHub/pydata/download/experiments'
    experiment_id = 7
    # tensile_specimen_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # tensile_specimen_ids = [1, 4, 7]
    tensile_specimen_ids = [11, 14, 17]
    # tensile_specimen_ids = [18, 23, 24]
    tensile_experiment_data, tensile_experiment_status = get_experiment_data(local_experiments_path, experiment_id, tensile_specimen_ids)
    processed_tensile_data = preproc_data(tensile_experiment_data, strain_shift=0.0)
    processed_tensile_data = partial_by_elastic_limit(processed_tensile_data)
    processed_tensile_data = reduce_to_target_rows(processed_tensile_data)


    experiment_id = 9
    # relax_specimen_ids = [1]
    relax_specimen_ids = [4]
    # relax_specimen_ids = [9]
    relax_experiment_data, relax_experiment_status = get_experiment_data(local_experiments_path, experiment_id, relax_specimen_ids)
    processed_relax_data = preproc_data(relax_experiment_data, strain_shift=0.0)
    processed_relax_data = reduce_to_target_rows(processed_relax_data)

    # paras_0 = [1.0, 2e-1, 1.0, 5.0e1, 1.0, 2.0e3]
    # constants = [0.97]

    paras_0 = [1, 1, 1]
    # constants = [0.95, 0.5, 40.0, 1500.0]
    constants = {'E_inf': 0.95,
                 'nu': 0.14,
                 'mode': 'analytical',
                 'tau_number': 3,
                 'tau': [0.05, 40.0, 1000.0]}

    # 0年
    # paras_0 = [0.01875972, 0.00206912]
    # paras = [0.01656102, 0.00236847]
    # constants = [0.95, 0.14, 7.49243691, 0.1, 5.92043313, 2.0, 1.37908231, 1000.0]
    # paras = [0.01875972, 0.00206912]
    # constants = [0.95, 0.14, 20.47216724, 0.05, 5.05709119, 40.0, 1.2055883, 1000.0]

    # 2年
    # paras_0 = [0.01992902, 0.00181885]
    # constants = [0.95, 0.14, 7.69765659, 0.1, 9.25720943, 2.0, 2.83577589, 1000.0]
    # constants = [0.95, 0.14, 25.79364856, 0.05, 8.41095917, 40.0, 2.39640248, 1000.0]

    # 8年
    # paras_0 = [0.02338096, 0.00357043]
    # constants = [0.95, 0.14, 17.65974303, 0.1, 6.20743176, 2.0, 4.36829172, 1000.0]
    # constants = [0.95, 0.14, 43.30986079, 0.05, 7.65646085, 40.0, 4.61372817, 1000.0]

    paras = fmin(func, paras_0, maxiter=5000, ftol=1e-4, xtol=1e-4, disp=True)
    print(paras)

    # for tau_1 in np.linspace(4e-1, 8e-1, 5):
    #     for tau_2 in np.linspace(1e1, 1e2, 5):
    #         for tau_3 in np.linspace(1e3, 2e3, 5):
    #             constants = [0.95, tau_1, tau_2, tau_3]
    #             paras = fmin(func, paras_0, maxiter=2000, ftol=1e-4, xtol=1e-4, disp=False)  # 优化后的参数
    #             print(tau_1, tau_2, tau_3, func(paras))

    plot_tensile(tensile_specimen_ids, processed_tensile_data, paras, constants)
    plot_relax(relax_specimen_ids, processed_relax_data, paras, constants)
