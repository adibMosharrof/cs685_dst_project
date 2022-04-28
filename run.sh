#! /bin/bash
source activate ./envs
time python trainer.py fit --config config/contrastive_osp.yaml 