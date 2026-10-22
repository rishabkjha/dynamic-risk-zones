40 crossings per scene, field of view 70°, optical flow off, robot max speed 0.8 m/s, zone sigmas 1, 2, 3.

| scene | method | success | collisions | robot at fault | person walked into robot | time to goal (s) | closest gap (m) | standing still (s) |
|---|---|---|---|---|---|---|---|---|
| eth | cv | 40% | 60% | 28% | 32% | 13.5 | -0.02 | 0.0 |
| eth | learned | 42% | 57% | 20% | 38% | 13.6 | -0.00 | 0.0 |
| hotel | cv | 72% | 28% | 10% | 18% | 13.8 | 0.21 | 0.0 |
| hotel | learned | 68% | 32% | 12% | 20% | 13.9 | 0.13 | 0.0 |
| univ | cv | 32% | 68% | 22% | 45% | 17.5 | -0.12 | 0.0 |
| univ | learned | 35% | 65% | 20% | 45% | 17.5 | -0.07 | 0.0 |
| zara1 | cv | 60% | 40% | 10% | 30% | 13.6 | 0.16 | 0.0 |
| zara1 | learned | 65% | 35% | 10% | 25% | 13.7 | 0.14 | 0.0 |
| zara2 | cv | 68% | 32% | 18% | 15% | 13.6 | 0.21 | 0.0 |
| zara2 | learned | 68% | 32% | 8% | 25% | 13.7 | 0.20 | 0.0 |
| **all** | **cv** | **54%** | **46%** | **18%** | **28%** | **14.4** | **0.09** | **0.0** |
| **all** | **learned** | **56%** | **44%** | **14%** | **30%** | **14.5** | **0.08** | **0.0** |

**Success per scene**

| scene | cv | learned |
|---|---|---|
| eth | 40% | 42% |
| hotel | 72% | 68% |
| univ | 32% | 35% |
| zara1 | 60% | 65% |
| zara2 | 68% | 68% |
