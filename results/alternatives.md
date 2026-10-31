Each method compared with the reference solution. Error is the largest error in body acceleration and tyre force, as a percentage of their peaks.

| Case | Method | Setting | Error | RHS evaluations | Run time [ms] | Memory [kB] |
|---|---|---|---|---|---|---|
| Speed hump, 75 mm x 3.7 m, 30 km/h | Euler, fixed step | h = 0.2 ms | 0.89% | 13720 | 103.3 | 874 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK4, fixed step | h = 10 ms | 0.53% | 1100 | 8.5 | 21 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK4, restarted at edges | h = 10 ms | 0.31% | 1100 | 9.3 | 25 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK45 adaptive | rtol = 0.001 | 0.26% | 800 | 10.9 | 82 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK45 adaptive, tight tolerance | rtol = 1e-06 | 0.00% | 1826 | 25.0 | 213 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK45 adaptive, restarted at edges | rtol = 0.001 | 0.19% | 708 | 10.1 | 82 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | Linear model, exact (lsim) | h = 1 ms | 0.00% | - | 9.1 | 389 |
| Pothole, 50 mm x 0.6 m, 40 km/h | Euler, fixed step | h = 0.1 ms | 0.88% | 23540 | 161.3 | 1313 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK4, fixed step | h = 1 ms | 0.89% | 9416 | 68.9 | 134 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK4, restarted at edges | h = 10 ms | 0.79% | 944 | 7.2 | 22 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK45 adaptive | rtol = 0.001 | 2.89% | 770 | 9.7 | 76 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK45 adaptive, tight tolerance | rtol = 1e-06 | 0.00% | 1970 | 24.6 | 210 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK45 adaptive, restarted at edges | rtol = 0.001 | 0.63% | 654 | 8.2 | 76 |
| Pothole, 50 mm x 0.6 m, 40 km/h | Linear model, exact (lsim) | h = 1 ms | 50.65% | - | 7.5 | 335 |
| Pothole, 50 mm x 0.6 m, 80 km/h | Euler, fixed step | h = 0.1 ms | 0.77% | 23270 | 164.1 | 1298 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK4, fixed step | h = 0.5 ms | 0.87% | 18616 | 115.7 | 262 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK4, restarted at edges | h = 10 ms | 0.58% | 932 | 6.5 | 22 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK45 adaptive | rtol = 0.001 | 10.22% | 668 | 8.2 | 72 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK45 adaptive, tight tolerance | rtol = 1e-06 | 100.00% | 92 | 1.1 | 13 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK45 adaptive, restarted at edges | rtol = 0.001 | 0.08% | 618 | 6.4 | 75 |
| Pothole, 50 mm x 0.6 m, 80 km/h | Linear model, exact (lsim) | h = 1 ms | 104.86% | - | 6.2 | 331 |
