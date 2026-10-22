Speed on **rishab-Vostro-3590** (x86_64, 8 threads), 320x240

| stage | ms per frame |
|---|---|
| stereo depth (SGBM) | 14.3 |
| optical flow (DIS) | 2.2 |
| detection (yolo11n_openvino_model) | 38.5 |
| prediction, 5 people (social GRU) | 2.5 |
| planning (91 candidate paths) | 1.0 |
| **total** | **58.6** (17.1 fps) |
