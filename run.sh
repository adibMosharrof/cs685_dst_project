#! /bin/bash
source activate ./envs
time python trainer.py fit --config config/slot_name_osp.yaml 