# Optimisation and performance

Case: default car, 50 mm deep, 0.6 m long pothole, 40 km/h (the wheel leaves the road). Run times are the fastest of 15 runs on this computer.

## 1. Profile of one run at the program's 1 ms step (cProfile, own time per function)

**Original code** (profiled total 248 ms):

| Function | Calls | Own time [ms] | Share |
|---|---|---|---|
| performance_study.py:original_pothole_profile | 9416 | 70.7 | 29% |
| performance_study.py:rhs | 9416 | 58.9 | 24% |
| solvers.py:rk4_step | 2354 | 45.8 | 19% |
| performance_study.py:r | 9416 | 27.9 | 11% |
| solvers.py:f_inside | 9416 | 15.9 | 6% |
| ~:<built-in method numpy.asarray> | 18836 | 11.0 | 4% |

**Improved code** (profiled total 160 ms):

| Function | Calls | Own time [ms] | Share |
|---|---|---|---|
| solvers.py:rk4_step | 2354 | 46.1 | 29% |
| model.py:rhs | 9416 | 32.7 | 20% |
| solvers.py:f_inside | 9416 | 16.8 | 11% |
| scenarios.py:r | 9416 | 13.5 | 8% |
| ~:<built-in method numpy.array> | 9417 | 10.6 | 7% |
| numeric.py:isscalar | 18832 | 8.8 | 5% |

## 2. Baseline against final design

Each step is the largest that keeps both the motion and the peak values within 1% (from section 4), except E, the 1 ms step the program uses.

| Version | Method | Code | Step | Motion error | Peak error | Steps | FLOPs per run | Run time [ms] | Achieved FLOP/s | Memory [kB] |
|---|---|---|---|---|---|---|---|---|---|---|
| A. Baseline | Fixed-step RK4 | original | 1 ms | 0.892% | 0.279% | 2354 | 537 k | 172.3 | 3.1 M | 96 |
| B. Code improved only | Fixed-step RK4 | improved | 1 ms | 0.892% | 0.279% | 2354 | 282 k | 70.5 | 4.0 M | 94 |
| C. Method improved only | RK4 restarted at edges | original | 2 ms | 0.021% | 0.248% | 1177 | 268 k | 90.3 | 3.0 M | 96 |
| D. Final, largest step within 1% | RK4 restarted at edges | improved | 2 ms | 0.021% | 0.248% | 1177 | 141 k | 39.6 | 3.6 M | 95 |
| E. Final, program setting | RK4 restarted at edges | improved | 1 ms | 0.003% | 0.061% | 2354 | 282 k | 78.8 | 3.6 M | 187 |

- Code: the improved code gives the same results as the original (largest difference 3.0e-16 of the largest state value, which is round-off) and is 2.4 times faster (A against B).
- Method: restarting at the road edges allows a 2 times larger step for the same accuracy, making it 1.9 times faster (A against C).
- Both together: 4.4 times faster than the baseline at the same accuracy (A against D).
- The program uses 1 ms (E) for smooth animation frames and sharper peaks: still 2.2 times faster than the baseline, with a much smaller error.

## 3. FLOPs and arithmetic intensity

| Code | FLOPs per right-hand side | FLOPs per RK4 step | Bytes per right-hand side | Arithmetic intensity [FLOP/byte] |
|---|---|---|---|---|
| original | 42 | 228 | 232 | 0.18 |
| improved | 15 | 120 | 112 | 0.13 |

- A 1000 x 1000 matrix product runs at 71.8 GFLOP/s on this computer, but one 4 x 4 matrix-vector product runs at only 14 MFLOP/s: for tiny arrays the time is set by the overhead of each NumPy call, not by the arithmetic.
- The simulation is in the same regime: the achieved FLOP/s in section 2 is thousands of times below what the computer can do. The data (about 100 to 250 bytes) stays in the fastest cache, so memory bandwidth does not limit it either. Speed-ups therefore come from doing fewer Python and NumPy operations, not fewer FLOPs.

## 4. Time step trade-off (error = larger of motion and peak error)

| Step | Baseline error | Baseline time [ms] | Final error | Final time [ms] |
|---|---|---|---|---|
| 10 ms | 18.374% | 17.2 | 10.618% | 7.9 |
| 5 ms | 2.299% | 34.4 | 1.572% | 15.9 |
| 2 ms | 1.685% | 88.3 | 0.248% | 38.9 |
| 1 ms | 0.892% | 174.1 | 0.061% | 79.4 |
| 0.5 ms | 0.458% | 342.0 | 0.006% | 156.1 |
| 0.2 ms | 0.361% | 863.9 | 0.003% | 391.3 |

## 5. Repeatability

- Results: two runs with the same inputs are identical.
- Run time over 30 runs: fastest 78.6 ms, median 80.6 ms, slowest 85.5 ms (spread 2% of the mean), which is why the fastest of several runs is reported.

## 6. Scalability (program setting, 1 ms steps)

| Simulated time [s] | Steps | Run time [ms] | Time per simulated second [ms] | Memory [kB] |
|---|---|---|---|---|
| 1.4 | 1354 | 44.7 | 33.0 | 109 |
| 2.4 | 2354 | 78.6 | 33.4 | 187 |
| 4.4 | 4354 | 145.0 | 33.3 | 344 |
| 8.4 | 8354 | 279.4 | 33.4 | 656 |
| 16.4 | 16354 | 548.5 | 33.5 | 1281 |

## 7. Numerical stability

- Largest stable RK4 step for this car: 39.5 ms (Euler: 7.0 ms). The program's 1 ms step is 40 times smaller, so accuracy, not stability, sets the step.
