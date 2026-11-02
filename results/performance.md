# Optimisation and performance

Case: default car, 50 mm deep, 0.6 m long pothole, 40 km/h (the wheel leaves the road). Run times are the fastest of 15 runs on this computer.

## 1. Profile of one run at the program's 1 ms step (cProfile, own time per function)

**Original code** (profiled total 182 ms):

| Function | Calls | Own time [ms] | Share |
|---|---|---|---|
| performance_study.py:original_pothole_profile | 9416 | 52.6 | 29% |
| performance_study.py:rhs | 9416 | 43.7 | 24% |
| solvers.py:rk4_step | 2354 | 33.0 | 18% |
| performance_study.py:r | 9416 | 20.3 | 11% |
| solvers.py:f_inside | 9416 | 11.5 | 6% |
| ~:<built-in method numpy.asarray> | 18836 | 8.1 | 4% |

**Improved code** (profiled total 119 ms):

| Function | Calls | Own time [ms] | Share |
|---|---|---|---|
| solvers.py:rk4_step | 2354 | 34.6 | 29% |
| model.py:rhs | 9416 | 23.7 | 20% |
| solvers.py:f_inside | 9416 | 12.3 | 10% |
| scenarios.py:r | 9416 | 10.1 | 8% |
| ~:<built-in method numpy.array> | 9417 | 7.8 | 7% |
| numeric.py:isscalar | 18832 | 6.6 | 6% |

## 2. Baseline against final design

Each step is the largest that keeps both the motion and the peak values within 1% (from section 4), except E, the 1 ms step the program uses.

| Version | Method | Code | Step | Motion error | Peak error | Steps | FLOPs per run | Run time [ms] | Achieved FLOP/s | Memory [kB] |
|---|---|---|---|---|---|---|---|---|---|---|
| A. Baseline | Fixed-step RK4 | original | 1 ms | 0.892% | 0.279% | 2354 | 537 k | 127.0 | 4.2 M | 96 |
| B. Code improved only | Fixed-step RK4 | improved | 1 ms | 0.892% | 0.279% | 2354 | 282 k | 51.3 | 5.5 M | 94 |
| C. Method improved only | RK4 restarted at edges | original | 2 ms | 0.021% | 0.248% | 1177 | 268 k | 66.0 | 4.1 M | 96 |
| D. Final, largest step within 1% | RK4 restarted at edges | improved | 2 ms | 0.021% | 0.248% | 1177 | 141 k | 30.2 | 4.7 M | 95 |
| E. Final, program setting | RK4 restarted at edges | improved | 1 ms | 0.003% | 0.061% | 2354 | 282 k | 60.0 | 4.7 M | 187 |

- Code: the improved code gives the same results as the original (largest difference 3.0e-16 of the largest state value, which is round-off) and is 2.5 times faster (A against B).
- Method: restarting at the road edges allows a 2 times larger step for the same accuracy, making it 1.9 times faster (A against C).
- Both together: 4.2 times faster than the baseline at the same accuracy (A against D).
- The program uses 1 ms (E) for smooth animation frames and sharper peaks: still 2.1 times faster than the baseline, with a much smaller error.

## 3. FLOPs and arithmetic intensity

| Code | FLOPs per right-hand side | FLOPs per RK4 step | Bytes per right-hand side | Arithmetic intensity [FLOP/byte] |
|---|---|---|---|---|
| original | 42 | 228 | 232 | 0.18 |
| improved | 15 | 120 | 112 | 0.13 |

- A 1000 x 1000 matrix product runs at 173.1 GFLOP/s on this computer, but one 4 x 4 matrix-vector product runs at only 18 MFLOP/s: for tiny arrays the time is set by the overhead of each NumPy call, not by the arithmetic.
- The simulation is in the same regime: the achieved FLOP/s in section 2 is thousands of times below what the computer can do. The data (about 100 to 250 bytes) stays in the fastest cache, so memory bandwidth does not limit it either. Speed-ups therefore come from doing fewer Python and NumPy operations, not fewer FLOPs.

## 4. Time step trade-off (error = larger of motion and peak error)

| Step | Baseline error | Baseline time [ms] | Final error | Final time [ms] |
|---|---|---|---|---|
| 10 ms | 18.374% | 12.8 | 10.618% | 6.0 |
| 5 ms | 2.299% | 25.8 | 1.572% | 12.1 |
| 2 ms | 1.685% | 63.7 | 0.248% | 30.8 |
| 1 ms | 0.892% | 124.2 | 0.061% | 57.3 |
| 0.5 ms | 0.458% | 252.6 | 0.006% | 115.6 |
| 0.2 ms | 0.361% | 675.0 | 0.003% | 287.1 |

## 5. Repeatability

- Results: two runs with the same inputs are identical.
- Run time over 30 runs: fastest 60.0 ms, median 61.7 ms, slowest 110.3 ms (spread 17% of the mean), which is why the fastest of several runs is reported.

## 6. Scalability (program setting, 1 ms steps)

| Simulated time [s] | Steps | Run time [ms] | Time per simulated second [ms] | Memory [kB] |
|---|---|---|---|---|
| 1.4 | 1354 | 34.5 | 25.5 | 109 |
| 2.4 | 2354 | 60.3 | 25.6 | 187 |
| 4.4 | 4354 | 112.8 | 25.9 | 344 |
| 8.4 | 8354 | 221.2 | 26.5 | 656 |
| 16.4 | 16354 | 450.7 | 27.6 | 1281 |

## 7. Numerical stability

- Largest stable RK4 step for this car: 39.5 ms (Euler: 7.0 ms). The program's 1 ms step is 40 times smaller, so accuracy, not stability, sets the step.
