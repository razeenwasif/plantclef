# Tests

This directory contains the `pytest` suite validating the structural, mathematical, and logical integrity of the Oracle system.

The tests ensure that:
- Core configuration schemas merge properly.
- The dynamic DALI dataloaders respect pre-split and stratified layouts.
- Post-processing algorithms (like Frank-Wolfe and Retinex normalization) are mathematically sound.
- The `oracle.py` CLI correctly dispatches commands to the launch scripts.

*Note: The CI pipeline (`.github/workflows/oracle_ci.yml`) runs these tests on every commit to prevent regressions.*