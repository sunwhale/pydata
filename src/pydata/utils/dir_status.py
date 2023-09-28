# -*- coding: utf-8 -*-
"""

"""
import json
import os
import time


def get_files(path):
    fns = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            fns.append([root, fn])
    return fns


def has_file(filename, files):
    b = False
    if filename in files:
        b = True
    return b


def format_size(size_in_bytes):
    try:
        size_in_bytes = float(size_in_bytes)
        kb = size_in_bytes / 1024
    except TypeError:
        print("传入的字节格式不对")
        return "Error"

    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M / 1024
            return "%.2fGB" % (G)
        else:
            return "%.2fMB" % (M)
    else:
        return "%.2fKB" % (kb)


def sub_dirs_int(path):
    try:
        sub_dirs = [int(sub_dir) for sub_dir in next(os.walk(path))[1]]
    except StopIteration:
        sub_dirs = []
    return sorted(sub_dirs)


def get_current_dir_int(path):
    abs_dir = next(os.walk(path))[0].split(os.sep)
    if abs_dir[-1] == '':
        return abs_dir[-2]
    else:
        return abs_dir[-1]


def sub_dirs(path):
    sub_dirs = next(os.walk(path))[1]
    return sub_dirs


def create_id(path):
    old_id_list = sub_dirs_int(path)
    if len(old_id_list) == 0:
        return 1
    else:
        return max(old_id_list) + 1


def files_in_dir(path):
    file_list = []
    for filename in sorted(next(os.walk(path))[2]):
        file = {}
        file['name'] = filename
        file['size'] = file_size(os.path.join(path, filename))
        file['time'] = file_time(os.path.join(path, filename))
        file_list.append(file)
    return file_list


def subpaths_in_dir(path):
    subpath_list = []
    for subpath_name in sorted(next(os.walk(path))[1]):
        subpath = {}
        subpath['name'] = subpath_name
        subpath['time'] = file_time(os.path.join(path, subpath_name))
        subpath_list.append(subpath)
    return subpath_list


def file_time(file):
    if os.path.exists(file):
        modified_time = os.path.getmtime(file)
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_time))
    else:
        return 'None'


def file_size(file):
    if os.path.exists(file):
        return format_size(os.path.getsize(file))
    else:
        return 'None'


def get_model_status(path, model_id):
    npy_file = os.path.join(path, str(model_id), 'model.npy')
    msg_file = os.path.join(path, str(model_id), 'model.msg')
    args_file = os.path.join(path, str(model_id), 'args.json')
    log_file = os.path.join(path, str(model_id), 'model.log')
    abaqus_file = os.path.join(path, str(model_id), 'abaqus', '.abaqus_msg')
    status = {}
    status['model_id'] = model_id
    try:
        with open(msg_file, 'r', encoding='utf-8') as f:
            message = json.load(f)
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
            status['args'] = str(args)
        with open(log_file, 'r', encoding='utf-8') as f:
            status['log'] = f.read()
        if os.path.exists(abaqus_file):
            with open(abaqus_file, 'r', encoding='utf-8') as f:
                abaqus_message = json.load(f)
            status['project_id'] = abaqus_message['project_id']
        else:
            status['project_id'] = 'None'
        status['npy_time'] = file_time(npy_file)
        status['npy_size'] = file_size(npy_file)
        status['size'] = str(message['size'])
        status['gap'] = message['gap']
        status['num_ball'] = message['num_ball']
        status['fraction'] = '%.4f' % message['fraction']
        status[
            'operation'] = "<a href='%s'>查看</a> | <a href='%s'>子模型</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
            '../view_model/' + str(model_id), '../manage_submodels/' + str(model_id),
            '../delete_models/' + str(model_id))
    except FileNotFoundError:
        for key in ['npy_time', 'npy_size', 'size', 'gap', 'num_ball', 'fraction', 'project_id']:
            status[key] = 'None'
        status['operation'] = "<a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                '../delete_models/' + str(model_id))
    return status


def get_submodel_status(path, model_id, submodel_id):
    npy_file = os.path.join(path, str(submodel_id), 'model.npy')
    msg_file = os.path.join(path, str(submodel_id), 'model.msg')
    args_file = os.path.join(path, str(submodel_id), 'args.json')
    status = {}
    status['model_id'] = model_id
    status['submodel_id'] = submodel_id
    try:
        with open(msg_file, 'r', encoding='utf-8') as f:
            message = json.load(f)
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
            status['args'] = str(args)
        status['npy_time'] = file_time(npy_file)
        status['npy_size'] = file_size(npy_file)
        status['ndiv'] = message['ndiv']
        status['location'] = str(message['location'])
        status['subsize'] = str(message['subsize'])
        status['gap'] = message['gap']
        status['num_ball'] = message['num_ball']
        status['fraction'] = '%.4f' % message['fraction']
        status['operation'] = "<a href='%s'>查看</a>" % ('../view_submodel/' + str(model_id) + '/' + str(submodel_id))
    except FileNotFoundError:
        for key in ['npy_time', 'npy_size', 'ndiv', 'location', 'subsize', 'gap', 'num_ball', 'fraction', 'operation']:
            status[key] = 'None'
    return status


def get_mesh_status(path, model_id, submodel_id):
    inp_file = os.path.join(path, str(submodel_id), 'Model-1.inp')
    args_file = os.path.join(path, str(submodel_id), 'args.json')
    msg_file = os.path.join(path, str(submodel_id), 'model.msg')
    status = {}
    status['model_id'] = model_id
    status['submodel_id'] = submodel_id
    if os.path.exists(inp_file):
        with open(args_file, 'r', encoding='utf-8') as f:
            args = json.load(f)
        with open(msg_file, 'r', encoding='utf-8') as f:
            message = json.load(f)
        status['args'] = str(args)
        status['gap'] = message['gap']
        status['nodes_number'] = message['nodes_number']
        status['elements_number'] = message['elements_number']
        status['node_shape'] = message['node_shape']
        status['dimension'] = message['dimension']
        status['size'] = message['size']
        status['element_type'] = message['element_type']
        status['inp_time'] = file_time(inp_file)
        status['inp_size'] = file_size(inp_file)
    return status


def get_doc_status(path, doc_id):
    md_file = os.path.join(path, str(doc_id), 'article.md')
    msg_file = os.path.join(path, str(doc_id), 'article.msg')
    status = {}
    if os.path.exists(md_file):
        status['doc_id'] = doc_id
        try:
            with open(msg_file, 'r', encoding='utf-8') as f:
                message = json.load(f)
            status['title'] = message['title']
            status['md_time'] = file_time(md_file)
            status[
                'operation'] = "<a href='%s'>查看</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                '../viewmd/' + str(doc_id), '../editmd/' + str(doc_id), '../deletemd/' + str(doc_id))
        except FileNotFoundError:
            for key in ['title', 'md_time']:
                status[key] = 'None'
            status[
                'operation'] = "<a href='%s'>查看</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                '../viewmd/' + str(doc_id), '../editmd/' + str(doc_id), '../deletemd/' + str(doc_id))
    return status


def get_sheet_status(path, sheet_id):
    sheet_file = os.path.join(path, str(sheet_id), 'sheet.json')
    msg_file = os.path.join(path, str(sheet_id), 'sheet.msg')
    status = {}
    if os.path.exists(sheet_file):
        status['sheet_id'] = sheet_id
        try:
            with open(msg_file, 'r', encoding='utf-8') as f:
                message = json.load(f)
            status['title'] = message['title']
            status['sheet_time'] = file_time(sheet_file)
            status[
                'operation'] = "<a href='%s'>查看</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                '../view_sheet/' + str(sheet_id), '../edit_sheet/' + str(sheet_id), '../delete_sheet/' + str(sheet_id))
        except FileNotFoundError:
            for key in ['title', 'sheet_time']:
                status[key] = 'None'
            status[
                'operation'] = "<a href='%s'>查看</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                '../view_sheet/' + str(sheet_id), '../edit_sheet/' + str(sheet_id), '../delete_sheet/' + str(sheet_id))
    return status


def get_project_status(path, project_id):
    msg_file = os.path.join(path, str(project_id), '.project_msg')
    status = {}
    if os.path.exists(msg_file):
        status['project_id'] = project_id
        try:
            with open(msg_file, 'r', encoding='utf-8') as f:
                message = json.load(f)
            status['name'] = message['name']
            status['project_time'] = file_time(msg_file)
            status['descript'] = message['descript']
            status['job'] = message['job']
            status['user'] = message['user']
            status['cpus'] = message['cpus']
            if 'link' in message.keys():
                status['link'] = message['link']
            else:
                status['link'] = 'None'
            status[
                'operation'] = "<a href='%s'>进入</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                'view_project/' + str(project_id), 'edit_project/' + str(project_id),
                'delete_project/' + str(project_id))
        except (FileNotFoundError, KeyError):
            for key in ['name', 'project_time', 'descript', 'job', 'user', 'cpus', 'link']:
                status[key] = 'None'
            status[
                'operation'] = "<a href='%s'>进入</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                'view_project/' + str(project_id), 'edit_project/' + str(project_id),
                'delete_project/' + str(project_id))
    return status


def get_virtual_status(path, virtual_id):
    msg_file = os.path.join(path, str(virtual_id), '.virtual_msg')
    status = {}
    if os.path.exists(msg_file):
        status['virtual_id'] = virtual_id
        try:
            with open(msg_file, 'r', encoding='utf-8') as f:
                message = json.load(f)
            status['name'] = message['name']
            status['template'] = message['template']
            status['virtual_time'] = file_time(msg_file)
            status['descript'] = message['descript']
            status['job'] = message['job']
            status['user'] = message['user']
            status['cpus'] = message['cpus']
            if 'link' in message.keys():
                status['link'] = message['link']
            else:
                status['link'] = 'None'
            status[
                'operation'] = "<a href='%s'>进入</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                'view_virtual/' + str(virtual_id), 'edit_virtual/' + str(virtual_id),
                'delete_virtual/' + str(virtual_id))
        except (FileNotFoundError, KeyError):
            for key in ['name', 'template', 'project_time', 'descript', 'job', 'user', 'cpus', 'link']:
                status[key] = 'None'
            status[
                'operation'] = "<a href='%s'>进入</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                'view_virtual/' + str(virtual_id), 'edit_virtual/' + str(virtual_id),
                'delete_virtual/' + str(virtual_id))
    return status


def get_experiment_status(path, experiment_id):
    msg_file = os.path.join(path, str(experiment_id), '.experiment_msg')
    status = {}
    if os.path.exists(msg_file):
        status['experiment_id'] = experiment_id
        try:
            with open(msg_file, 'r', encoding='utf-8') as f:
                message = json.load(f)
            status['name'] = message['name']
            status['type'] = message['type']
            status['material'] = message['material']
            status['standard'] = message['standard']
            status['experiment_time'] = file_time(msg_file)
            status['descript'] = message['descript']
            if 'link' in message.keys():
                status['link'] = message['link']
            else:
                status['link'] = 'None'
            status[
                'operation'] = "<a href='%s'>进入</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                'view_experiment/' + str(experiment_id), 'edit_experiment/' + str(experiment_id),
                'delete_experiment/' + str(experiment_id))
        except (FileNotFoundError, KeyError):
            for key in ['name', 'experiment_time', 'type', 'material', 'standard', 'descript']:
                status[key] = 'None'
            status[
                'operation'] = "<a href='%s'>进入</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                'view_experiment/' + str(experiment_id), 'edit_experiment/' + str(experiment_id),
                'delete_experiment/' + str(experiment_id))
    return status


def get_specimen_status(path, experiment_id, specimen_id):
    msg_file = os.path.join(path, str(experiment_id), str(specimen_id), '.specimen_msg')
    para_json_file = os.path.join(
        path, str(experiment_id), str(specimen_id), 'parameters.json')
    status = {}
    status['experiment_id'] = experiment_id
    status['specimen_id'] = specimen_id
    try:
        with open(msg_file, 'r', encoding='utf-8') as f:
            message = json.load(f)
        if os.path.exists(para_json_file):
            with open(para_json_file, 'r', encoding='utf-8') as f:
                parameters = json.load(f)
        else:
            parameters = {}
        for p in parameters.keys():
            status[p] = parameters[p]
        status['name'] = message['name']
        status['descript'] = message['descript']
        status['specimen_time'] = file_time(msg_file)
        status['path'] = os.path.join(path, str(experiment_id), str(specimen_id))
        button = ""
        button += "<a class='btn btn-primary btn-sm' href='%s'>查看</a> " % (
                '../view_specimen/' + str(experiment_id) + '/' + str(specimen_id))
        button += "<a class='btn btn-primary btn-sm' onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a> " % (
                '../delete_specimen/' + str(experiment_id) + '/' + str(specimen_id))
        status['operation'] = button
    except FileNotFoundError:
        for key in ['name', 'descript', 'specimen_time', 'path']:
            status[key] = 'None'
        status['operation'] = "<a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                '../delete_specimen/' + str(experiment_id) + '/' + str(specimen_id))
    return status


def get_template_status(path, template_id):
    msg_file = os.path.join(path, str(template_id), '.template_msg')
    status = {}
    if os.path.exists(msg_file):
        status['template_id'] = template_id
        try:
            with open(msg_file, 'r', encoding='utf-8') as f:
                message = json.load(f)
            status['name'] = message['name']
            status['template_time'] = file_time(msg_file)
            status['descript'] = message['descript']
            status['job'] = message['job']
            status['user'] = message['user']
            status['cpus'] = message['cpus']
            if 'link' in message.keys():
                status['link'] = message['link']
            else:
                status['link'] = 'None'
            status[
                'operation'] = "<a href='%s'>进入</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模板?')\" href='%s'>删除</a>" % (
                'view_template/' + str(template_id), 'edit_template/' + str(template_id),
                'delete_template/' + str(template_id))
        except (FileNotFoundError, KeyError):
            for key in ['name', 'template_time', 'descript', 'job', 'user', 'cpus', 'link']:
                status[key] = 'None'
            status[
                'operation'] = "<a href='%s'>进入</a> | <a href='%s'>编辑</a> | <a onclick=\"return confirm('确定删除模板?')\" href='%s'>删除</a>" % (
                'view_template/' + str(template_id), 'edit_template/' + str(template_id),
                'delete_template/' + str(template_id))
    return status


def get_job_status(path, project_id, job_id):
    msg_file = os.path.join(path, str(project_id), str(job_id), '.job_msg')
    para_json_file = os.path.join(
        path, str(project_id), str(job_id), 'parameters.json')
    status = {}
    status['project_id'] = project_id
    status['job_id'] = job_id
    try:
        with open(msg_file, 'r', encoding='utf-8') as f:
            message = json.load(f)
        if os.path.exists(para_json_file):
            with open(para_json_file, 'r', encoding='utf-8') as f:
                parameters = json.load(f)
        else:
            parameters = {}
        for p in parameters.keys():
            status[p] = parameters[p]

        solver_status_file = os.path.join(
            path, str(project_id), str(job_id), '.solver_status')
        if os.path.exists(solver_status_file):
            with open(solver_status_file, 'r', encoding='utf-8') as f:
                solver_status = f.read()
        else:
            solver_status = 'None'

        prescan_status_file = os.path.join(
            path, str(project_id), str(job_id), '.prescan_status')
        if os.path.exists(prescan_status_file):
            with open(prescan_status_file, 'r', encoding='utf-8') as f:
                prescan_status = f.read()
        else:
            prescan_status = 'None'

        odb_to_npz_status_file = os.path.join(
            path, str(project_id), str(job_id), '.odb_to_npz_status')
        if os.path.exists(odb_to_npz_status_file):
            with open(odb_to_npz_status_file, 'r', encoding='utf-8') as f:
                odb_to_npz_status = f.read()
        else:
            odb_to_npz_status = 'None'

        odb_to_npz_proc_file = os.path.join(
            path, str(project_id), str(job_id), '.odb_to_npz_proc')
        if os.path.exists(odb_to_npz_proc_file):
            with open(odb_to_npz_proc_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) != 0:
                    odb_to_npz_proc = lines[-1]
                else:
                    odb_to_npz_proc = '0.0'
        else:
            odb_to_npz_proc = ''

        job_name = message['job']
        inp_file = os.path.join(path, str(project_id), str(job_id), '%s.inp' % job_name)
        odb_file = os.path.join(path, str(project_id), str(job_id), '%s.odb' % job_name)
        npz_file = os.path.join(path, str(project_id), str(job_id), '%s.npz' % job_name)
        status['job'] = message['job']
        status['user'] = message['user']
        status['cpus'] = message['cpus']
        status['inp_time'] = file_time(inp_file)
        status['inp_size'] = file_size(inp_file)
        status['odb_time'] = file_time(odb_file)
        status['odb_size'] = file_size(odb_file)
        status['npz_time'] = file_time(npz_file)
        status['npz_size'] = file_size(npz_file)
        status['solver_status'] = solver_status
        status['prescan_status'] = prescan_status
        status['odb_to_npz_status'] = odb_to_npz_status
        status['odb_to_npz_proc'] = odb_to_npz_proc
        status['parameters'] = str(parameters)
        status['path'] = os.path.join(path, str(project_id), str(job_id))
        button = ""
        button += "<a class='btn btn-primary btn-sm' href='%s'>查看</a> " % (
                '../view_job/' + str(project_id) + '/' + str(job_id))
        button += "<a class='btn btn-primary btn-sm' onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a> " % (
                '../delete_job/' + str(project_id) + '/' + str(job_id))
        if solver_status == 'Submitting' or solver_status == 'Running' or solver_status == 'Pause' or solver_status == 'Stopping':
            button += "<button class='btn btn-secondary btn-sm' disabled='disabled'>计算</button> "
        else:
            button += "<a href='%s' class='btn btn-success btn-sm'>计算</a> " % (
                    '../run_job/' + str(project_id) + '/' + str(job_id))
        # if solver_status=='Running':
        #     button += "<a href='#' class='btn btn-warning btn-sm'>暂停</a> "
        # else:
        #     button += "<button class='btn btn-secondary btn-sm' disabled='disabled'>暂停</button> "
        # if solver_status=='Pause':
        #     button += "<a href='#'' class='btn btn-info btn-sm'>继续</a> "
        # else:
        #     button += "<button class='btn btn-secondary btn-sm' disabled='disabled'>继续</button> "
        if solver_status == 'Running':
            button += "<a href='%s' class='btn btn-danger btn-sm'>终止</a> " % (
                    '../terminate_job/' + str(project_id) + '/' + str(job_id))
        else:
            button += "<button class='btn btn-secondary btn-sm' disabled='disabled'>终止</button> "
        if prescan_status != 'Scanning' and prescan_status != 'Submitting':
            button += "<a class='btn btn-success btn-sm' href='%s'>扫描</a> " % (
                    '../prescan_odb/' + str(project_id) + '/' + str(job_id))
        else:
            button += "<button class='btn btn-secondary btn-sm' disabled='disabled'>扫描</button> "
        if odb_to_npz_status != 'Running' and odb_to_npz_status != 'Submitting':
            button += "<a class='btn btn-success btn-sm' href='%s'>导出</a> " % (
                    '../odb_to_npz/' + str(project_id) + '/' + str(job_id))
        else:
            button += "<button class='btn btn-secondary btn-sm' disabled='disabled'>导出</button> "
        status['operation'] = button
    except FileNotFoundError:
        for key in ['job', 'user', 'cpus']:
            status[key] = 'None'
        status['operation'] = "<a onclick=\"return confirm('确定删除模型?')\" href='%s'>删除</a>" % (
                '../delete_job/' + str(project_id) + '/' + str(job_id))
    return status


def packing_models_detail(path):
    data_list = []
    model_id_list = sub_dirs_int(path)
    for model_id in model_id_list:
        status = get_model_status(path, model_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


def packing_submodels_detail(path, model_id):
    data_list = []
    submodel_id_list = sub_dirs_int(path)
    for submodel_id in submodel_id_list:
        status = get_submodel_status(path, model_id, submodel_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


def docs_detail(path):
    data_list = []
    doc_id_list = sub_dirs_int(path)
    for doc_id in doc_id_list:
        status = get_doc_status(path, doc_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


def experiments_detail(path):
    data_list = []
    experiment_id_list = sub_dirs_int(path)
    for experiment_id in experiment_id_list:
        status = get_experiment_status(path, experiment_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


def experiment_specimens_detail(path, experiment_id):
    data_list = []
    specimen_id_list = sub_dirs_int(os.path.join(path, str(experiment_id)))
    for specimen_id in specimen_id_list:
        status = get_specimen_status(path, experiment_id, specimen_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


def sheets_detail(path):
    data_list = []
    sheet_id_list = sub_dirs_int(path)
    for sheet_id in sheet_id_list:
        status = get_sheet_status(path, sheet_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


def projects_detail(path):
    data_list = []
    project_id_list = sub_dirs_int(path)
    for project_id in project_id_list:
        status = get_project_status(path, project_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


def project_jobs_detail(path, project_id):
    data_list = []
    job_id_list = sub_dirs_int(os.path.join(path, str(project_id)))
    for job_id in job_id_list:
        status = get_job_status(path, project_id, job_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


def virtuals_detail(path):
    data_list = []
    virtual_id_list = sub_dirs_int(path)
    for virtual_id in virtual_id_list:
        status = get_virtual_status(path, virtual_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


def templates_detail(path):
    data_list = []
    template_id_list = sub_dirs_int(path)
    for template_id in template_id_list:
        status = get_template_status(path, template_id)
        data_list.append(status)

    data = {
        "data": data_list
    }

    return data


if __name__ == '__main__':
    path = r'F:/GitHub/pydata/download/experiments'
    print(experiment_specimens_detail(path, 7))
