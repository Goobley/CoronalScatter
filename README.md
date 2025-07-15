## Coronal Scattering Code

Accelerated program for calculating radio scattering in the solar corona, using Kokkos to abstract parallelism, based on the original IDL scattering code developed by E. Kontar, present as `ray_new.pro`.

## Requirements

- HDF5
- netCDF4
- Kokkos

There are scripts to install these in the `docker_deps` directory, and all of them can be vendored into the tree with a bit of tweaking.

Other dependencies:

- YAKL
- argparse for C++
- fmt

Also builds on code developed for DexRT.

## Building

This program uses CMake for configuration. Example scripts that configure the CMake setup are present in `build/`, and these are used to produce a Makefile (or similar).

## Running

The program is configured using command line options. To see these, run `./scatter --help`, which should output something along the lines of:

```
Usage: Scatter [--help] [--version] [--eps EPS] [--r-init R_INIT] [--theta0 THETA0] [--f-ratio F_RATIO] [--aniso ANISO] [--asym ASYM] [--seed SEED] [--n-particles N_PARTICLES] [--dt-save dt_save] [--filename FILE]

Optional arguments:
  -h, --help     shows help message and exits
  -v, --version  prints version information and exits
  --eps          Scattering epsilon. [nargs=0..1] [default: 0.1]
  --r-init       initial position (R_sun). Sets plasma frequency. [nargs=0..1] [default: 1.75]
  --theta0       Initial angle (degrees). [nargs=0..1] [default: 0]
  --f-ratio      Ratio of emission to plasma frequency. 2.0 for harmonic. [nargs=0..1] [default: 1.1]
  --aniso        Density fluctuation anisotropy, aniso = q_parallel / q_perp_x. 1 => isotropic, 0 => 2D density fluctuations, -> infty => quasi-parallel fluctuations. [nargs=0..1] [default: 0.3]
  --asym         Density fluctuation asymmetry along r. 1 => symmetric, 0 < asym < 1 => more outward density fluctuation (stronger scattering inwards), 1 < asym < 2 => more inward density fluctuation (stronger scattering outwards). [nargs=0..1] [default: 1]
  --seed         Random seed for simulation. [nargs=0..1] [default: 110081]
  --n-particles  Number of particles to simulate. [nargs=0..1] [default: 1048576]
  --dt-save      Output timestep (set to -1 to calculate internally). [nargs=0..1] [default: -1]
  --filename     Name for output file [nargs=0..1] [default: "scatter.nc"]

Scattering for radio emission in the solar corona, using Kokkos.
```

As such, an invocation of `scatter` could look like: `./scatter --eps 0.4 --r-init 1.5 --theta0 15.0 --f-ratio 2.0 --asym 1.5 --dt-save 0.1 --filename my_scatter_model.nc`.
