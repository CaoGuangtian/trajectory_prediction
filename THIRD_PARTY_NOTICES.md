# Third-party notices

## PatchTST

The trajectory PatchTST architecture in this project was informed by the official implementation of “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers” by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam.

- Upstream project: <https://github.com/yuqinie98/PatchTST>
- Local reference inspected during development: `D:\code\RealTimeTrajectory\reference\patchTST`
- Upstream license: Apache License 2.0

The project-specific model, data windowing, split isolation, training loop, and evaluation integration were implemented separately for this repository. The upstream dataset splitting, trajectory smoothing, and experiment loop were not reused.
