Each method compared with the reference solution. Error is the largest error in body acceleration and tyre force, as a percentage of their peaks.

| Case | Method | Setting | Error | RHS evaluations | Run time [ms] | Memory [kB] |
|---|---|---|---|---|---|---|
| Speed hump, 75 mm x 3.7 m, 30 km/h | Euler, fixed step | h = 0.2 ms | 0.89% | 13720 | 74.2 | 874 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK4, fixed step | h = 10 ms | 0.53% | 1100 | 6.5 | 20 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK4, restarted at edges | h = 10 ms | 0.31% | 1100 | 7.2 | 25 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK45 adaptive | rtol = 0.001 | 0.26% | 800 | 9.3 | 81 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK45 adaptive, tight tolerance | rtol = 1e-06 | 0.00% | 1826 | 21.4 | 213 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | RK45 adaptive, restarted at edges | rtol = 0.001 | 0.19% | 708 | 8.5 | 82 |
| Speed hump, 75 mm x 3.7 m, 30 km/h | Linear model, exact (lsim) | h = 1 ms | 0.00% | - | 19.3 | 390 |
| Pothole, 50 mm x 0.6 m, 40 km/h | Euler, fixed step | h = 0.1 ms | 0.88% | 23540 | 125.4 | 1313 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK4, fixed step | h = 1 ms | 0.89% | 9416 | 55.1 | 134 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK4, restarted at edges | h = 10 ms | 0.79% | 944 | 7.0 | 22 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK45 adaptive | rtol = 0.001 | 2.89% | 770 | 9.3 | 76 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK45 adaptive, tight tolerance | rtol = 1e-06 | 0.01% | 1970 | 24.4 | 208 |
| Pothole, 50 mm x 0.6 m, 40 km/h | RK45 adaptive, restarted at edges | rtol = 0.001 | 0.63% | 654 | 8.3 | 76 |
| Pothole, 50 mm x 0.6 m, 40 km/h | Linear model, exact (lsim) | h = 1 ms | 50.65% | - | 17.9 | 335 |
| Pothole, 50 mm x 0.6 m, 80 km/h | Euler, fixed step | h = 0.1 ms | 0.77% | 23270 | 160.4 | 1298 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK4, fixed step | h = 0.5 ms | 0.87% | 18616 | 148.2 | 262 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK4, restarted at edges | h = 10 ms | 0.58% | 932 | 8.1 | 22 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK45 adaptive | rtol = 0.001 | 10.22% | 668 | 10.4 | 72 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK45 adaptive, tight tolerance | rtol = 1e-06 | 100.00% | 92 | 1.6 | 13 |
| Pothole, 50 mm x 0.6 m, 80 km/h | RK45 adaptive, restarted at edges | rtol = 0.001 | 0.08% | 618 | 9.9 | 75 |
| Pothole, 50 mm x 0.6 m, 80 km/h | Linear model, exact (lsim) | h = 1 ms | 104.86% | - | 22.3 | 331 |
| Sharp bump, 150 mm x 0.3 m, 110 km/h, soft car | Euler, fixed step | h = 0.02 ms (finest tried) | 3.29% | 115491 | 847.7 | 7334 |
| Sharp bump, 150 mm x 0.3 m, 110 km/h, soft car | RK4, fixed step | h = 1 ms | 0.48% | 9240 | 70.7 | 149 |
| Sharp bump, 150 mm x 0.3 m, 110 km/h, soft car | RK4, restarted at edges | h = 2 ms | 0.10% | 4620 | 40.0 | 94 |
| Sharp bump, 150 mm x 0.3 m, 110 km/h, soft car | RK45 adaptive | rtol = 0.001 | 100.00% | 50 | 1.6 | 10 |
| Sharp bump, 150 mm x 0.3 m, 110 km/h, soft car | RK45 adaptive, tight tolerance | rtol = 1e-06 | 100.00% | 50 | 0.9 | 10 |
| Sharp bump, 150 mm x 0.3 m, 110 km/h, soft car | RK45 adaptive, restarted at edges | rtol = 0.001 | 6.52% | 708 | 11.1 | 93 |
| Sharp bump, 150 mm x 0.3 m, 110 km/h, soft car | Linear model, exact (lsim) | h = 1 ms | 161.86% | - | 22.3 | 329 |
