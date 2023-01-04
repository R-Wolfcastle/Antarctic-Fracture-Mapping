#!/bin/bash

export_vars () {
    MODELA_DICT_DIR="../models/a/"
    MODELB_DICT_DIR="../models/b/"
    MODULE_HOME="../pub_modules/"
    SCRIPT_HOME="../"
    ASF_UNAME=""
    ASF_PWD=""
    DEM_PAR_FILEPATH=""
    DEM_FILEPATH=""
    ORBIT_DIR=""
    CLUSTER_CONDA_CONFIG=""
}

set -a
export_vars
set -o allexport
