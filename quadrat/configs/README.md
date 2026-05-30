# Configs

This directory contains the YAML configuration files for the plantclef pipeline.

These files dictate hyperparameter settings, dataset bindings, hardware behavior (like FP8 and low-RAM modes), and model architectures for the different stages of the system (Foundation Caching, Expert Specialization, Student Distillation, and Inference). 

The `datasets.yaml` file acts as the central registry for dynamically discovered datasets, supporting both "flat" and "pre_split" directory layouts.