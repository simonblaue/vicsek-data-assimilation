# Viszeck model analyses with Kalman Filter


## The Viszeck model

Parameters from paper:

$\Delta t$ = 1

$0.003 < v < 0.3$ no big difference 

$v = 0.03$

Interaction radius $r=1$


Variational Parameters:

$L = 7$
$N = 300$
$\eta = 2.0$


## ToDOs

- ~~weigh own direction much stronger too still get individual agents and long trajectories~~
- ~~maybe gaussian noise instead of uniform in viscek~~
- Measure Kalman does it equilibrate?, if so average over error after equilibration to compare against different config 
- velocity variations
- shuffled observations
- 


## What to Test:

-  Ensemble size
-  measurement noise
-  with vs without angel observation
-  

## How to test

- 10 seeds, 200 steps
- agents: 50/100
- ensembles: 10, 50, 100, 200
- observation noise 0.0001 - 1, 5 values logspace
- optional: observable axis (all vs x,y only)
- 2, 4

## tmux

https://gist.github.com/henrik/1967800

