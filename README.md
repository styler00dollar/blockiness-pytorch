# blockiness-pytorch

A pytorch implemenation of [gohtanii/DiverSeg-dataset](https://github.com/gohtanii/DiverSeg-dataset/blob/main/blockiness/calc_blockiness.py) which is faster than the original. Inference on cpu is faster than on gpu.

| Code             | Time                 | Blockiness         |
| ---------------- | -------------------- | ------------------ |
| torch cpu (fp32) | 0.23279643058776855s | 3.5402135848999023 |
| torch gpu (fp32) | 0.8458788394927979s  | 3.540212392807007  |
| numpy            | 2.2084133625030518s  | 3.540212368790306  |

