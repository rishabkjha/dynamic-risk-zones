Stereo distance vs tape measure, 22 Oct 2026. Calibration: baseline 110.9 mm (ruler ~108 mm), focal 262.8 px at 320x240, stereo RMS 0.373 px. Target: the calibration board / a printed box, distance taped from the lenses, median depth in a 13x13 px square.

| true distance (m) | raw stereo (m) | raw error | corrected (m) | corrected error |
|---|---|---|---|---|
| 1.0 | 0.940 | -6.0 cm (-6.0%) | 0.992 | -0.8 cm (-0.8%) |
| 2.0 | 1.807 | -19.3 cm (-9.7%) | 2.009 | +0.9 cm (+0.4%) |
| 3.0 | 2.619 | -38.1 cm (-12.7%) | 3.065 | +6.5 cm (+2.2%) |
| 4.0 | 3.237 | -76.3 cm (-19.1%) | 3.948 | -5.2 cm (-1.3%) |
| 5.0 | 3.951 | -104.9 cm (-21.0%) | 5.063 | +6.3 cm (+1.3%) |

Raw stereo always reads short, and the error grows with distance. In disparity terms every reading
is too large by about the same amount (1.86, 1.56, 1.41, 1.72, 1.55 px at 320x240):
a constant disparity offset, which a small rotation left between the rectified cameras produces
(1.62 px here is 3.2 px at 640x480, about 0.35 degrees of yaw). It is not a code error: the
rectified principal points are identical (cx = 328.86 px in both P1 and P2), and the 30 mm board
gives a baseline within 3 mm of the ruler, which would only cause a constant ~2.7% scale error.

"Corrected" subtracts DISPARITY_OFFSET_PX = 1.62 px (config.py) from every disparity. The offset
is the mean of the five excesses above, i.e. **fitted on these same five points**, so the corrected
column shows how well one constant explains the error (within 7 cm up to 5 m), not an independent
test. A fresh run of 11_depth_check.py with the correction on gives the independent check.
