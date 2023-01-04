#!/bin/bash

export_vars () {
    MODULE_HOME="../pub_modules/"
    SCRIPT_HOME="../"
    CLUSTER_CONDA_CONFIG=""
    DATA_SUPERDIR=""
}

set -a
export_vars
set -o allexport
