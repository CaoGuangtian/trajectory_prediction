# Third-party notices

## PatchTST

The trajectory PatchTST backbone in this project is adapted from the official implementation of “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers” by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam.

- Upstream project: <https://github.com/yuqinie98/PatchTST>
- Local reference inspected during development: `D:\code\RealTimeTrajectory\reference\patchTST`
- Upstream license: Apache License 2.0; a copy is included at `licenses/PatchTST-APACHE-2.0.txt`

The adapted backbone retains patching, channel-independent encoding, residual attention, normalization, RevIN, positional encoding, and the flatten prediction head. It has been modified to consume this project's typed configuration and local-ENU tensor interface.

The trajectory data windowing, aircraft-level split isolation, train-only normalization, masked loss, training loop, and evaluation integration are project-specific. The upstream dataset splitting, trajectory smoothing, and experiment loop were not reused.
