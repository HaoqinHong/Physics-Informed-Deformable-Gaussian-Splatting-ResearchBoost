# Environment and publication portability

The recorded runs used one NVIDIA A800-SXM4-80GB, Python 3.12, CUDA 12.8,
`torch.__version__ = 2.7.0a0+ecf3bae40a.nv25.02`, and torchvision `0.22.0a0`.
Package metadata normalizes the torch suffix to `nv25.2`. Additional measured
Python versions are pinned in `requirements-repro.txt`; `cv2` must be available.
Use an isolated environment and a compatible PyTorch/CUDA build; the published
requirements do not replace PyTorch or install packages into an existing shared environment.

A matching NVIDIA release suffix alone does not prove a matching image or
PyTorch commit. For example, the [official NVIDIA25.02 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-02.html)
list a different torch source commit. This repository makes no claim that a
public container tag reconstructs the exact original binary environment.
Rebuild extensions, record versions and hashes, collect fresh reference arms,
and freeze fresh numerical thresholds before judging a new environment.

Publication changes are deployment portability changes: PIDG's GPU UUID is
loaded from its generated config; 3DGS's optional viewer registration is loaded
from its generated config, with no viewer allowed by default. Configs must be
created before baseline formulation and remain frozen afterward. The numerical
judge, threshold algorithms and timing decisions were retained. These portable
harness files consequently have different hashes from the historical frozen
deployment; historical certificate/manifest hashes cannot validate a new checkout.
The historical numeric outcomes are preserved as data, including failures.

The PIDG restored source was recovered from the public Python3.8 bytecode in the
pinned upstream repository and validated by function bytecode/constant comparison.
Publication removes only decompiler path/time header comments. The two standard
header includes and GLM build-include compatibility edits are shared by the new
reference and candidate. The accepted acceleration changes only flow gather
backward; rejected raster patches are opt-in diagnostics.

No data, trained checkpoints, compiled extensions, deployment paths, process IDs,
GPU UUIDs or credentials are committed. Public data manifests pin downloaded
bytes. Full original large traces/reference tensors remain in the original
experiment storage; new users regenerate them with the provided harness.
