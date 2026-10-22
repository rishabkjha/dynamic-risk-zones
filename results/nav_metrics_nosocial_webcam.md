40 crossings per scene, field of view 70°, optical flow on, robot max speed 0.8 m/s, zone sigmas 1, 2, 3.

| scene | method | success | collisions | robot at fault | person walked into robot | time to goal (s) | closest gap (m) | standing still (s) |
|---|---|---|---|---|---|---|---|---|
| eth | learned_nosocial | 42% | 57% | 30% | 28% | 13.6 | 0.02 | 0.0 |
| hotel | learned_nosocial | 68% | 32% | 12% | 20% | 13.9 | 0.17 | 0.0 |
| univ | learned_nosocial | 28% | 72% | 25% | 48% | 17.9 | -0.12 | 0.0 |
| zara1 | learned_nosocial | 65% | 35% | 15% | 20% | 13.7 | 0.16 | 0.0 |
| zara2 | learned_nosocial | 75% | 25% | 8% | 18% | 13.7 | 0.25 | 0.0 |
| **all** | **learned_nosocial** | **55%** | **44%** | **18%** | **26%** | **14.6** | **0.09** | **0.0** |

**Success per scene**

| scene | learned_nosocial |
|---|---|
| eth | 42% |
| hotel | 68% |
| univ | 28% |
| zara1 | 65% |
| zara2 | 75% |
