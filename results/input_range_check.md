Program time step h = 1 ms (RK4 restarted at edges) at the extremes of every input range: 16 cars (every combination of the lowest and highest body mass, spring stiffness, damping ratio and tyre stiffness) over 6 extreme roads. Runs with both the motion and the peaks within 1% of the reference: 95 of 96. Worst error: 1.20%.

The five worst runs:

| Body mass [kg] | Spring [kN/m] | Damping ratio | Tyre [kN/m] | Road | Speed [km/h] | Motion error | Peak error |
|---|---|---|---|---|---|---|---|
| 700 | 5 | 0.05 | 400 | 150 mm x 0.3 m bump | 110 | 1.20% | 0.68% |
| 700 | 100 | 1 | 400 | 150 mm x 0.3 m bump | 110 | 0.02% | 0.92% |
| 150 | 100 | 1 | 100 | 150 mm x 0.3 m bump | 110 | 0.00% | 0.87% |
| 150 | 5 | 0.05 | 400 | 150 mm x 0.3 m bump | 110 | 0.54% | 0.75% |
| 150 | 100 | 0.05 | 400 | 150 mm x 0.3 m bump | 110 | 0.47% | 0.73% |
