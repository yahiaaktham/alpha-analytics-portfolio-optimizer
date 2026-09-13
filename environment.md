# Environment used to generate these results

Pinning `requirements.txt` is necessary but not sufficient to reproduce every
figure exactly. See report Section 18.4: the unregularised mean-variance
baseline depends on the platform's BLAS/LAPACK kernel selection. Every
regularised strategy and every Monte Carlo risk metric is platform-independent.

## Platform

- Operating system: Windows-11-10.0.26100-SP0
- Machine: AMD64
- Python: 3.13.14

## Packages

- numpy: 2.4.1
- pandas: 3.0.0
- scipy: 1.17.0
- sklearn: 1.8.0
- matplotlib: 3.11.1

## Linear algebra backend

- blas: scipy-openblas 0.3.30
- lapack: scipy-openblas 0.3.30
