---
layout: page
title: A Landau Level Simulator
img: assets/img/landaulevel/icon.png
importance: 3
category: research
---
You may find a blog titled ["Reconciling Different Viewpoints of Landau Level Fan Chart"]
(/blog/2022/landaulevel-simulator/) describing more
details alongside this exploration.

## Introduction
A Landau Level fan chart is a graphical representation used in the study of quantum Hall effects, depicting the relationship between energy levels (Landau levels) and magnetic field strength in one (or more) two-dimensional electron system.
### A theoretic picture of Landau level fan
This is a collection of all Landau levels originating from different electronic bands in an energy-versus-magnetic field strength plot. 
This can directly come from theoretic calculations.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/landaulevel/model3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
### An experimental Landau level fan
 In experimental measurements, the variation of energy is actually realized via controlling the population of electrons (electron density) as tuning the voltage applied to one side of a capacitor placed on top of a device (gate).  Importantly, the highest energy of populated electrons (chemical potential) is not linearly dependent on the electron density.  However, this fan chart is always compared to a theoretical calculation of Landau levels in energy versus magnetic field strength. This difference (nonlinear transition) between theoretic picture and physical realization always causes disputes between theorists and experimentalists.
## Implementation
The source code of this open-source project is hosted by [Github](https://github.com/LarsonLaugh/toyband).
### Design
The design of this project can be broken down into four steps: Firstly, construct an electronic band structure comprised
of multiple single bands. These bands need to be defined by setting up parameters, e.g., choice of band (linear or quadratic), 
effective mass for quadratic bands, Fermi velocity for linear bands and zero-field energy point for each band.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/landaulevel/model2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/landaulevel/model1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
Secondly, we calculate the Landau level splitting as function of magnetic field. Thirdly, we translate this energy versus field into 
a density versus field plot.  This process involves calculating the total density at each energy level (a loose definition of
density of state), in another word, constructing an one-to-one correspondence between energy and density. The last step is
to collect this transformed results of Landau levels at the chemical potential position in a density-versus-field plot for
a series of discrete density values corresponding to experimentally attainable range. The collection of results will be then ready
for a comparison with experiments (Landau fan chart).
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/landaulevel/llsimulator.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
