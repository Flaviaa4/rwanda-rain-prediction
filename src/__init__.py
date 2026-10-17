"""Rwanda next-day rain prediction pipeline.

The package is organised as one connected pipeline in which every module receives the
outputs of the previous stage as arguments:

    config -> data_download -> data_prep -> features -> split
           -> baselines / ml / dl (orchestrated by runner) -> evaluation -> plots

All settings come from ``configs/config.yaml`` and ``configs/experiments.yaml``, so the
notebooks only call functions and display their outputs.
"""
