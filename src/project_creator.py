#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def project_creator(path, exist_ok):
    if exist_ok:
        project_path = 'project/project'
    else:
        n = 1
        project_path = f'project/project{n}'
        while os.path.isdir(project_path):
            n += 1
            project_path = f'project/project{n}'

    project_path = os.path.join(path, project_path)
    os.makedirs(project_path, exist_ok=True)
    return project_path
