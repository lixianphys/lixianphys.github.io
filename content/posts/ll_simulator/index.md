---
author: ["Lixian Wang"]
title: A Simulator for Landau Level Fan Chart 
tags: ["Landau level", "quantum Hall effect"]
categories: ["physics", "algorithm"]
date: "2022-07-15"
ShowToc: true
TocOpen: true
draft: false
---

## Background
### Landau level
In solid-state physics, the famous [band theory](https://en.wikipedia.org/wiki/Electronic_band_structure) describes how 
electrons are distributed favorably over energy and momentum in electronic materials. The states (energy, momentum, spin
and etc.) of electrons collectively form an electronic band structure. When subjecting to a magnetic field, these electronic
states would shift in energy according to its interactive terms with the field. As this field gets very strong, the interactive
term (orbital contribution) becomes dominant and start to form dense bands with narrow energy dispersion (A lot of states
have almost the same energy). These bands are also energetically discrete and referred to as ["Landau level"](https://en.wikipedia.org/wiki/Landau_quantization).

### Experimental Landau fans and different perspectives of theorists
If scanning the Landau levels in energy (shift the observation window energetically in y-direction) for a series of
increasing magnetic fields (x-direction), we would be able to revealing the energy information of various Landau levels
and how they evolve with the field. However, in experimental measurements, this scan is not so straightforward to implement,
because that we don't control the energy of observation window (chemical potential) directly. The variation of energy is
actually realized via controlling the population of electrons (electron density) as tuning the voltage applied to one side
of a capacitor placed on top of a device (gate).  Importantly, the highest energy of populated electrons (chemical potential)
is not linearly dependent on the electron density. This difference (nonlinear transition) between theoretic picture and
physical realization always causes disputes between theorists and experimentalists. 

---

### An experimental attempt: A Landau-level simulator
To win our theorist friends and keep ourselves happy, we need to connect our experimental data (Landau levels evolves in
electron density) with more favorable physical picture in theorist's mind (Landau levels evolve in energy).  A more
challenging task is to interpret our experimental data and translate it into an electronic band structure. A simulator
of Landau levels is thus aimed to provide theoretic prediction that is directly comparable to experiments for a predefined
electronic band structure at zero magnetic field. If this prediction aligns with experiments well, we would be able to
conclude the electronic band structure in an error-and-try manner. At worst, this Landau level simulator will hopefully
provide some insights into making choice between multiple reasonable guesses. 


---

## Overview
The design of this project can be broken down into four steps: Firstly, construct an electronic band structure comprised 
of multiple single bands. These bands need to be defined by setting up parameters, e.g., choice of band (linear or quadratic), 
effective mass for quadratic bands, Fermi velocity for linear bands and zero-field energy point for each band.  Secondly, 
we calculate the Landau level splitting as function of magnetic field. Thirdly, we translate this energy versus field into 
a density versus field plot.  This process involves calculating the total density at each energy level (a loose definition of 
density of state), in another word, constructing an one-to-one correspondence between energy and density.  The last step is 
to collect this transformed results of Landau levels at the chemical potential position in a density-versus-field plot for 
a series of discrete density values corresponding to experimentally attainable range. The collection of results will be then ready
for a comparison with experiments (Landau fan chart).

---

## Key aspects of a CLI software
The source code of this open-source project is hosted in [Github](https://github.com/LarsonLaugh/toyband)
### Electronic band structure
The electronic band structure (energy versus momentum) of materials can be defined before entering actual calculations. 
For simplicity, it is assumed to be isotropic. Though this assumption seems to be oversimplified, we still can reach 
amazingly good match with experiments.  This indicates that decorating the electron band by adding more terms and symmetries
to its Hamiltonian is not as crucial as expected at least in the material system we studied. 
### Configuration
Default settings such IO, plotting styles and some modelling parameters (Landau level broadening) are stored in `config.py`.
### Functional descriptions
**Add a band to the system**

```shell
python addband.py [-density] {-is_dirac} {-is_cond} [-gfactor GFACTOR] [{-dp VF D} {-cp MEFF SPIN}]
```
**Remove band(s) from the system**
```shell
python delband.py [-i INDEX or all]
```
**Snapshot the system**
```shell
python peeksys.py
```
**Landau levels manifested in energy versus field**
```shell
python run.py [-enplot] {-dir DIR} {-fnm FNM} [--enrange Estart Eend Enum] [--bfrange Bstart Bend Bnum] {-nmax NMAX} {-angle ANGLE}
```
**Landau levels manifested in density versus field**
```shell
python run.py [-denplot] {-dir DIR} {-fnm FNM} [--enrange Estart Eend Enum] [--bfrange Bstart Bend Bnum] {-nmax NMAX} {-angle ANGLE}
```
**Simulation of Landau fan chart for given densities**
 01. Plot the density versus field relationship for a set of given densities (specified by `--allden` and `-nos`) as a simulation of Landau fan chart.
```shell 
python run.py [-simu] [--allden "NS1 NE1 NS2 NE2 ..."] [-nos NOS] {-dir DIR} {-fnm FNM} [--enrange Estart Eend Enum] [--bfrange Bstart Bend Bnum] {-nmax NMAX} {-angle ANGLE}
```
02. Alternatively, one can load densities stored in a csv file (no header) under a directory ``DIR`` and with a filename ``FNM`` :
```shell
python run.py [-simu] [-loadden path-to-csvfile] {-dir DIR} {-fnm FNM} [--enrange Estart Eend Enum] [--bfrange Bstart Bend Bnum] {-nmax NMAX} {-angle ANGLE}
```

**Density of state (DOS)**
01. Plot the DOS versus field relationship at a fixed chemical potential
```shell
python run.py [-dos] {-dir DIR} {-fnm FNM} [--enrange Estart Eend Enum] [--bfrange Bstart Bend Bnum] {-nmax NMAX} {-angle ANGLE}
```
02. Plot the DOS as function of density and field
 ```shell
python run.py [-dosm] [-loadden path-to-csvfile] {-dir DIR} {-fnm FNM} [--enrange Estart Eend Enum] [--bfrange Bstart Bend Bnum] {-nmax NMAX} {-angle ANGLE}
```
03. Plot the data from csv file
```shell
python plotfile.py [-f path-to-csvfile]
```
---

### Modules
**addband**                                       
`python addband.py [-option][parameters]`
Add a new band into the system. A system will be automatically initiated upon the creation of the first band.

| Option     | Description  |
| ------------- | ----------------------------------------------------------- |
| `density`     | density for this band in units of 1/m^2 |
| `-is_cond`    | is a conduction band or not |
| `-gfactor`    | g-factor for this band |
| `-dp DP1 DP2` | For linearly dispersing bands. DP1: fermi velocity (in units of m/s) DP2: D (meV nm^2) for a Dirac-like band. D(kx^2+ky^2) accounts for the deviation from a linear Dirac dispersion.   |
| `-cp CP1 CP2` | For quadratically dispersing bands. CP1: absolute value of effective mass (in units of rest mass of electron) CP2: spin number (+1 or -1 or 0) for a conventional band. The spin number stands for spin-up (+1), spin-down (-1) and spinless (0), respectively |

**peeksys**
`python peaksys.py`
Peek into the system you created. Once you'd like to see what is in your system, how many bands and their parameters. The above command will print out a summary of bands in table-like format.

**delband**
`python delband.py`
Remove bands from a system

It will prompt up a dialogue `which band to delete? Input the index number: Or press 'e' to exit`. Add the number in front of all the parameters within a row for that band you want to delete. Or quit by typing `e`.  In some cases, you need to delete bands in multiple-line scripts, then just using `python toybands/delband.py -i [INDEX or all]` 'all' will remove all. 
***module `run`
 `python run.py [-option][parameters]`
Carry out various tasks in the existing material system.

| Option      | Description                                                                                 |
| ----------- | ----------------------------------------------------- |
| `--enrange` | energy range: start end NumofPoints   |
| `--bfrange` | magnetic field range: start end NumofPoints           |
| `-enplot`   | plot the energy versus bfield                                       |
| `-denplot`  | plot the density versus bfield             |
| `-simu`     | simulation                                                                        |
| `-dosm`     | mapping of DOS onto (n,B)                                            |
| `-dos`      | plot dos versus bfield                                                     |
| `--allden`  | densities for each band: start1 end1 start2 end2 ...  |
| `-nos`      | number of steps in the simulation    |
| `-dir`      | relative output directory                                                |
| `-fnm`    | filename                                                                          |
| `-nmax`  | number of Landau levels to be considered (default=20) |
| `-angle` | angle in degree made with the sample plane norm by the external field (default=0) |
| `-scond` | set Landau level broadening parameter sigma at B = 1 T for conduction bands       |
| `-sval`   | set Landau level broadening parameter sigma at B = 1 T for valence bands     |
| `-zll`    | reduce the density of state (DOS) of zero LLs to its half compared to other LLs   |