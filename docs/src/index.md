```@meta
CurrentModule = Lehmann
```
Documentation for [Lehmann](https://github.com/numericaleft/Lehmann.jl).

# Discrete Lehmann Representation (DLR)

This package provides subroutines to represent and manuipulate Green's functions in the imaginary-time or in the Matsubara-frequency domain. 

Imaginary-time Green's functions encode the thermodynamic properites of quantum many-body systems. At low temperature, they are typically very singular and hard to deal with in numerical calculations. 

The physical Green's functions always have the analytic structure specified by the Lehmann representation,
$$G(\tau)=-\int_{-\infty}^{\infty} K(\tau, \omega) \rho(\omega) d \omega$$
where $\tau$ is the imaginary time, $\omega$ is the real frequency. While the spectral density $\rho(\omega)$ depends on the details of the quantum many-body system, the convolution kernel $K(\tau, \omega)$ is universal and is roughly an exponential function $\exp(-\omega \tau)$. 

If one cares about the thermodynamic quantities, one only needs to manipulate the Green's functions. Then DLR allows us to represent the Green's function up to an accuracy $\epsilon$ with a fake spectral function only has a handful poles,
$$
G(\tau) \approx G_{\mathrm{DLR}}(\tau) \equiv \sum_{k=1}^{r} K\left(\tau, \omega_{k}\right) \widehat{\rho}_{k},
$$
where $r$ is called the rank of DLR. It is of the order,
$$
r \sim \log \frac{E_{uv}}{T} \log \frac{1}{ϵ}
$$

where $T$ is the temperature, $E_{uv}$ is the ultraviolet energy scale beyond which the physical spectral function decays away, $\epsilon$ is the accuracy.

# Main Features

We provide the following components to ease the numerical manipulation of the Green's functions:

- Algorithms to generate the discrete Lehamnn representation (DLR), which is a generic and compact representation of Green's functions proposed in the Ref. [1]. In this package, two algorithms are provided: one algorithm is based on conventional QR algorithm, another is based on a functional QR algorithm. The latter extends DLR to extremely low temperature.

- Dedicated DLR for Green's functions with the particle-hole symmetry (e.g. phonon propagator) or with the particle-hole antisymmetry (e.g. superconductor gap function).

- Fast and accurate Fourier transform between the imaginary-time domain and the Matsubara-frequency domain with a cost $\sim O(log(1/T)log(1/ϵ))$ and an accuracy ~100ϵ.

- Fast and accurate Green's function interpolation with a cost $\sim O(log(1/T)log(1/ϵ))$ and an accuracy ~100ϵ.

# Reference

If this library helps you to create software or publications, please let us know, and cite

[1] ["Discrete Lehmann representation of imaginary time Green's functions", Jason Kaye, Kun Chen, and Olivier Parcollet, arXiv:2107.13094](https://arxiv.org/abs/2107.13094)

[2] ["libdlr: Efficient imaginary time calculations using the discrete Lehmann representation", Jason Kaye and Hugo U.R. Strand, arXiv:2110.06765](https://arxiv.org/abs/2110.06765)

<!-- ```@index
```

```@contents
Pages = [
    "../../README.md"
]
Depth = 1
``` -->
