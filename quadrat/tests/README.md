# Tests

This directory contains the `pytest` suite validating the structural, mathematical, and logical integrity of the plantclef system.

The tests ensure that:
- Core configuration schemas merge properly.
- The dynamic DALI dataloaders respect pre-split and stratified layouts.
- Post-processing algorithms (like Frank-Wolfe and Retinex normalization) are mathematically sound.
- The `plantclef.py` CLI correctly dispatches commands to the launch scripts.

*Note: The CI pipeline (the GitHub Actions CI) runs these tests on every commit to prevent regressions.*