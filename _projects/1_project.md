---
layout: page 
title: Interacting Counter-propagating Flows
description: algorithms aiming at solving all interactions
img: assets/img/interact_counter_flow/icon.png 
importance: 1
category: research
---
You may find a blog titled ["A Duo Traveling Reversely"](/blog/2023/counterflow/) describing more details
alongside this exploration.

We describe a methodology that addresses the consequence of sequential interactions between counter-propagating flows.
Though being sparsely discussed, the nature of interaction between counter-propagating flows is fundamentally distinct
from that between co-propagating flows, featured by bidirectional, convoluted information flow. To the best of our
knowledge, a complete picture of sequential interactions between counter-propagating flows is still lacking.

We develop a set of algorithms that are responsible for breaking this convoluted interaction process down into solvable
sub-problems. The ultimate goal is to achieve the transition matrix connecting initial and final states of
counter-propagating flows, and also intermediate states in between interactions, in turn visualizing the evolution of
flows via interactions.

In particular, a low-level algorithm, referred to as “squeeze", is to calculate the transition matrix connecting initial
states and final states for a minimum sequence containing two interactions. To calculate a longer sequence, this squeeze
algorithm would be invoked repeatedly to take new interactions into account as moving forwardly in a sequence, referred
to as forward-propagation algorithm. Lastly, another (backward-propagation) algorithm acts backwardly to calculate
intermediate states. In condensed matter physics, this methodology extends the application of Landauer-Büttiker 
formalism into the domain of highly-correlated edge transport.

### Algorithm Design

Here all three algorithms will be introduced individually, which collectively find out the matrix representation of
combined effect of sequential interactions and all intermediate states.

#### Squeeze Algorithm

This algorithm is named for the fact the calculation is carried out simultaneously from both ends towards the middle of
the sequence of interactions, resembling the action of squeezing. In short, this algorithm comprises three major steps:
a. Connect initial states with the intermediate states; b. Connect intermediate states with the final states; c. Connect
initial states and the final states via the connections established in steps a) and b). This algorithm acts as a
low-level agent and will be invoked repeatedly.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/interact_counter_flow/squeeze_algorithm.png" class="img-fluid rounded 
z-depth-1" zoomable=true %}
    </div>
</div>

#### Forward-propagation Algorithm

This algorithm starts with the head of the sequence, and then keeps adding following interactions into the calculation
process, pushing the calculation forwardly till reaching the last interaction at the end of the sequence. In each
iteration, this high-level algorithm will invoke aforementioned "squeeze method" for adding a subsequent interaction.
Ultimately, this algorithm would yield the matrix representation of sub-sequences (from the first to the interaction in
process) and a series of hidden parameters necessary for calculating intermediate states later.

#### Backward-propagation Algorithm

This algorithm is aimed to derive all intermediate states between interactions backwardly. Employing this algorithm
requires matrix representations and hidden parameters passed by the forward-propagation process.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/interact_counter_flow/highlevel_algorithm.png" class="img-fluid rounded 
z-depth-1" zoomable=true %}
    </div>
</div>

### Making Use of Algorithms in Condensed Matter Physics

#### Quantum Hall Effect and Edge-state Picture

When a two-dimensional (extremely thin film), highly-conducting (high crystalline quality) semiconductor-like material (
e.g., GaAs heterostructure) is subject to a vertically aligned magnetic field (approx. 6 orders of magnitude stronger
than Earth's magnetic field), a profound and intriguing phenomenon, referred to
as [quantum Hall effect](https://en.wikipedia.org/wiki/Quantum_Hall_effect)., appears and has been extensively studied
since its birth in 1980. In history, several Noble Prizes have been awarded to related discoveries
in [1985](https://www.nobelprize.org/prizes/physics/1985/press-release/)
, [1998](https://www.nobelprize.org/prizes/physics/1998/press-release/)
, [2016](https://www.nobelprize.org/prizes/physics/2016/summary/). Aside from its deeper physical implication
like [topology in physics](https://pubs.aip.org/physicstoday/article/56/8/38/388340/A-Topological-Look-at-the-Quantum-Hall-EffectThe)
, a well-celebrated, trustful "edge-state" physical picture for this phenomenon is that, electrons are forced to move
alongside the thin film edges in one direction, forming dissipationless "edge states" (electron flows). Without delving
into deeper theory, a "magic" force seems to guide the motion of electrons. Nevertheless, this simple picture is
accurate enough to be applicable in academic discussions and exciting enough to spur expectations of revolution in the
field of electric power transmission if no magnetic field is needed (See [quantum spin Hall effect](https://en.wikipedia.org/wiki/Quantum_spin_Hall_effect)).

#### Landauer-Büttiker Formalism and Interaction between Edge States

Back to "edge states", this unique "bullet-like" (ballistic) motion of electrons can be most suitably described by a
model developed by Markus Büttiker in early 80s based
on [Rolf Landauer's earlier work](https://en.wikipedia.org/wiki/Landauer_formula), later referred to
as [Landauer-Büttiker formalism](https://de.wikipedia.org/wiki/Landauer-B%C3%BCttiker-Formalismus#:~:text=Der%20Landauer%2DB%C3%BCttiker%2DFormalismus%2C,Quanten%2DHall%2DEffekts%20verwendet.)
. Often we do not assume these edge states interact with each other or exchange electrons. In a typical device with
terminals (electron reservoirs), interactions between edge states are in principle not discernible if all edge states
move in the same direction (a co-propagating mode). This is because that co-propagating edge states themselves are
identical in terms of physical properties like electrochemical potential and any exchange between them would not lead to
any observable consequence. In order to detect this type of interaction, dedicated device design is needed to divert
co-propagating edge states into different paths. For counter-propagating edge states, this interaction is directly measurable 
even in a standard device (uniform path). 

#### Realize Individual Paths of Edge States

I would like to highlight the implementation of dynamic programming, which allows for diverting edge states into
different paths.

In a standard Landauer-Büttiker system, all edge states enter each terminal without exception. The electronic
measurement is only performed via terminals which measure all edge states at once. Consequently, this measurement result
is an average of all edge states in terms of electro-chemical potential.

To extract additional information of edge states, it is advisable to block some of them from particular terminals,
leading to different paths. In other words, this additional controls imposed on edge states may reveal hidden
information, not detectable in conventional configuration (standard Landauer-Büttiker system). However, a complex
blockage pattern may also cause difficulty in defining the blocked edge state, because that the
information carried by these edge states, not detectable by terminals, spread into other edge states via interactions 
directly or indirectly. An algorithm is developed to resolve this complexity recursively.

### Code implementation

The source code of this open-source project is hosted in [Github](https://github.com/LarsonLaugh/Counterfusion). 

The design principle for usage is to facilitate the bottom-up construction of a system from the building block -
interaction. Here is a snapshot to illustrate how easy it is to build a system and visualize the relevant results.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/interact_counter_flow/demo.jpg" class="img-fluid rounded z-depth-1" 
zoomable=true %}
    </div>
</div>

