---
layout: page
title: A Genetic-algorithm Optimized PID Controller
description: algorithm trained with simulated data from a physical model
img: assets/img/pidcontroller/icon_pid.png
importance: 2
category: fun
---

You may find a blog titled ["How to Turn My Kettle into a Sous Vide Cooker"](/blog/2022/pid/) describing more 
details alongside this exploration.

This personal project is to refactor my old kettle with a PID controller and its accessories to regulate water temperature below the boiling point like a Sous vide cooker.  The line of thought on implementation is described in this chart:
<div class="col-sm mt-3 mt-md-0">  
    {% include figure.html path="assets/img/pidcontroller/PID workflow.png" class="img-fluid rounded z-depth-1" zoomable=true %}  
</div> 
Let's first go through some basics.

## Basics
### What Is a PID Controller?
A PID (Proportional-Integral-Derivative) controller is a widely used control loop feedback mechanism in industrial control systems. They are extensively utilized in processes where precise and stable control is essential, such as in temperature regulation in chemical plants, speed control in conveyor systems, and pressure control in oil refineries. Furthermore, PID controllers play a crucial role in automation, enhancing efficiency, and consistency in manufacturing processes, and are integral in emerging technologies within robotics and unmanned vehicles, where precise control is paramount. Their adaptability and accuracy make them an indispensable tool in the modern industrial landscape.
### How Does a PID Controller Function?
At its core, a PID controller continuously calculates an error value as the difference between a desired setpoint and a measured process variable (temperature), subsequently applying a correction based on proportional, integral, and derivative terms. The proportional term determines the reaction to the current error, the integral term assesses the accumulation of past errors, and the derivative term predicts future errors, providing a smoother and more precise control method. This sophisticated balancing act makes PID controllers exceptionally versatile and effective in various industrial applications. 

### How to Apply Genetic Algorithm to This PID Controller?

Genetic Algorithms (GAs) are a type of evolutionary algorithm inspired by the process of natural selection, designed to solve complex optimization problems. In the context of a PID (Proportional-Integral-Derivative) controller, GAs can be particularly effective. They work by encoding the PID controller's parameters (proportional, integral, and derivative gains) into a format akin to chromosomes, which are then subjected to processes mimicking biological evolution, such as selection, crossover, and mutation. Through iterative rounds of these processes, the genetic algorithm evolves a population of solutions, progressively improving the PID parameters to optimize the controller's performance. This method is especially useful in complex systems where the optimal PID settings are not easily determinable due to nonlinearities or interactions between multiple variables. 

## Implementation
### 01. Physical Modeling  
Heat transfer can take place in four different ways : [Advection, thermal conduction, convection and radiation](https://en.wikipedia.org/wiki/Heat_transfer). Here we only consider thermal conduction as the major contributor. We also consider metallic walls of the kettle to be perfectly transparent in terms of thermal conduction. Therefore, the hot water directly interfaces the cold air outside. In short, we estimate that the temperature goes up at a rate of **0.3 deg/sec** when the heater is on and decreases at a rate of **0.012 deg/sec** when off. Note that this estimate is not a function of temperature. 
### 02. Algorithm Training Process 
The source code of this open-source project is hosted by [Github](https://github.com/LarsonLaugh/pidML) .The same algorithm has been realized in both Python (`pidtrain.py`) and C++ (`pidtrain.cpp`). For comparison, C++ code is 50 times faster than its Python counterpart. 
### 03. Deployment of the Microcontroller  
The brain of the PID controller is a microcontroller (popular choices like Pi Pico and Ardurio Nano). A temperature 
sensor 
is inserted into the water to continuously monitor the temperature. And a relay is installed on the power line to execute on/off command from the microcontroller.  For convenience, an user interface for manually setting up PID parameters is also implemented by a ICD 1602 screen and a rotary potentiometer.  Below is a list of used hardware components:

| Functional parts  | Models                          |  
|-------------------|---------------------------------|  
| Microcontroller   | Raspberry Pi Pico/ Ardurio Nano |  
| Temperature sensor | MAX6675                         |  
| 220V control      | Relay                           |  
| User interface    | ICD 1602                        |  
| User input        | rotary potentiometer            |  

Here is a diagram for wiring them: 
<div class="col-sm mt-3 mt-md-0">  
    {% include figure.html path="assets/img/pidcontroller/sketch_ArduinoNano.png" class="img-fluid rounded z-depth-1" zoomable=true %}  
</div>  
  
The deploying code has also been realized in both Micropython and Ardurio for Pico and Ardurio Nano, respectively. The `Micropython` code for microcontroller unit (MCU) is stored in `mpython` folder.  The Arduino code for  MCU is stored in `Arduino` folder. 

### 04. Performance
<div class="col-sm mt-3 mt-md-0">  
    {% include figure.html path="assets/img/pidcontroller/realvssimu.png" class="img-fluid rounded z-depth-1" zoomable=true %}  
</div>  
Here we show the actual temperature curve (labelled as "real") and the simulation result (labelled as "simulation") with the same PID parameters.  In the first 2000 seconds, the simulation matches quite well with the actual curve in both trend of temperature (upper) and calculated output (`Kp*error+Ki*integral+Kd*derivative`) of the PID feedback loop (lower). That means our physical model estimates accurately the change rate of temperature used for training process. After 2000 seconds, the simulation attains a stabilization of temperature within 0.32 degree. In reality, the temperature continues to fluctuate in a narrowed window ~ 5 degrees in recorded time. This fluctuation seems to be overshooting, which may be due to the fact that the heater is still much hotter than water and continues to heat water even after being powered off. This may be the bottleneck to achieve better performance.   