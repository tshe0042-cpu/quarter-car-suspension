Cost of following the motion within 1% on the medium pothole (50 mm deep, 0.6 m long). Adaptive runs are checked every 0.1 ms using their dense output.

| Method | Speed | Time step or tolerance | Error | RHS evaluations |
|---|---|---|---|---|
| Euler | 40 km/h | h = 0.1 ms | 0.88% | 23540 |
| RK4 | 40 km/h | h = 1 ms | 0.89% | 9416 |
| RK4, restarted at edges | 40 km/h | h = 10 ms | 0.79% | 944 |
| RK45 (adaptive) | 40 km/h | rtol = 0.001 | 2.89% | 770 |
| RK45 (adaptive) | 40 km/h | rtol = 1e-06 | 0.00% | 1970 |
| RK45 (adaptive), restarted at edges | 40 km/h | rtol = 0.001 | 0.63% | 654 |
| RK45 (adaptive), restarted at edges | 40 km/h | rtol = 1e-06 | 0.00% | 1524 |
| Euler | 80 km/h | h = 0.1 ms | 0.77% | 23270 |
| RK4 | 80 km/h | h = 0.5 ms | 0.87% | 18616 |
| RK4, restarted at edges | 80 km/h | h = 10 ms | 0.58% | 932 |
| RK45 (adaptive) | 80 km/h | rtol = 0.001 | 10.22% | 668 |
| RK45 (adaptive) | 80 km/h | rtol = 1e-06 | 100.00% | 92 |
| RK45 (adaptive), restarted at edges | 80 km/h | rtol = 0.001 | 0.08% | 618 |
| RK45 (adaptive), restarted at edges | 80 km/h | rtol = 1e-06 | 0.00% | 1398 |
