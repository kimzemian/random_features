# fast data driven control

## Description
designed data driven gaussian process controllers for 6 different gaussian process methods: 

1,2. affine dense kernel and corresponding affine dense random fourier features
3,4. affine dot product kernel and corresponding affine dot product random fourier features
5,6. vanilla kernel method and vanilla random fourier features

plots and experiments implemented for all controllers on inverted pendulum.

work in progress, more to come!

## Setup
To initialize the submodule dynamics library:
```
git submodule update --init --recursive
```
Then to setup
```
cd core; python setup.py develop

cd fast_control; python -e setup.py
```
## Usage

episodic usage
```
gps_names options are ad_kernel, adp_kernel, ad_rf, adp_rf
```