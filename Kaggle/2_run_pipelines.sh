#!/bin/bash
########################################################################################################################
# 2_run_pipelines.sh - Runs the train and predict for each dataset to train and then generate predictions
########################################################################################################################

########################################################################################################################
# Data - Is Epic Intro
########################################################################################################################
python train.py -d "resources/externaldata/data/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "resources/pretrained/models/Is Epic/"
python predict.py -d "resources/externaldata/data/Data - Is Epic Intro 2024-03-25" -l "Is Epic Files.txt" -t "Is Epic" -m "resources/pretrained/models/Is Epic/" -o "resources/externaldata/predictions/Is Epic Intro Full.csv"

########################################################################################################################
# Data - Needs Respray
########################################################################################################################
python train.py -d "resources/externaldata/data/Data - Needs Respray - 2024-03-26" -l "Labels-NeedsRespray-2024-03-26.csv" -t "Needs Respray" -o "resources/pretrained/models/Needs Respray/"
python predict.py -d "resources/externaldata/data/Data - Needs Respray - 2024-03-26" -l "Needs Respray.txt" -t "Needs Respray" -m "resources/pretrained/models/Needs Respray/" -o "resources/externaldata/predictions/Needs Respray Full.csv"

########################################################################################################################
# Data - Is GenAI
########################################################################################################################
python train.py -d "resources/externaldata/data/Data - Is GenAI - 2024-03-25" -l "Labels-IsGenAI-2024-03-25.csv" -t "Is GenAI" -o "resources/pretrained/models/Is GenAI/"
python predict.py -d "resources/externaldata/data/Data - Is GenAI - 2024-03-25" -l "Is GenAI Files.txt" -t "Is GenAI" -m "resources/pretrained/models/Is GenAI/" -o "resources/externaldata/predictions/Is GenAI Full.csv"

