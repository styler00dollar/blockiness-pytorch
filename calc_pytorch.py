# https://github.com/gohtanii/DiverSeg-dataset/tree/main/blockiness
import os
import time
import argparse

import numpy as np
import cv2
import torch
from tqdm import tqdm
from basicsr.utils import scandir
from typing import Tuple

BLOCK_SIZE = 8
device = torch.device("cpu")


class DCT:
    """
    Discrete Cosine Transform (DCT) class.

    Original code reference:
    https://gist.github.com/TonyMooori/661a2da7cbb389f0a99c
    """

    def __init__(self, N=BLOCK_SIZE):
        self.N = N
        self.phi_1d = torch.zeros((N, N), device=device)
        for k in range(N):
            if k == 0:
                self.phi_1d[k] = torch.ones(N, device=device) / torch.sqrt(
                    torch.tensor(N, dtype=torch.float32)
                )
            else:
                n_range = torch.arange(N, dtype=torch.float32, device=device)
                self.phi_1d[k] = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(
                    (k * torch.pi / (2 * N)) * (n_range * 2 + 1)
                )

        self.phi_2d = torch.zeros((N, N, N, N), device=device)
        for i in range(N):
            for j in range(N):
                self.phi_2d[i, j] = torch.outer(self.phi_1d[i], self.phi_1d[j])

    def dct2(self, data):
        data = torch.from_numpy(data).float().to(device)
        reshaped_data = data.reshape(self.N * self.N)
        reshaped_phi_2d = self.phi_2d.reshape(self.N * self.N, self.N * self.N)
        dct_result = torch.matmul(reshaped_phi_2d, reshaped_data)
        return dct_result.reshape(self.N, self.N)


def calc_margin(height, width):
    h_margin = height % BLOCK_SIZE
    w_margin = width % BLOCK_SIZE
    cal_height = height - (h_margin if h_margin >= 4 else h_margin + BLOCK_SIZE)
    cal_width = width - (w_margin if w_margin >= 4 else w_margin + BLOCK_SIZE)
    h_margin = (h_margin + BLOCK_SIZE) if h_margin < 4 else h_margin
    w_margin = (w_margin + BLOCK_SIZE) if w_margin < 4 else w_margin
    return cal_height, cal_width, h_margin, w_margin


def calc_DCT(img, dct, h_block_num, w_block_num):
    dct_img = torch.zeros(
        (h_block_num * BLOCK_SIZE, w_block_num * BLOCK_SIZE), device=device
    )
    batch_size = 64
    blocks = []
    positions = []

    for h_block in range(h_block_num):
        for w_block in range(w_block_num):
            block = img[
                h_block * BLOCK_SIZE : (h_block + 1) * BLOCK_SIZE,
                w_block * BLOCK_SIZE : (w_block + 1) * BLOCK_SIZE,
            ]
            blocks.append(block)
            positions.append((h_block, w_block))

            if len(blocks) == batch_size or (
                h_block == h_block_num - 1 and w_block == w_block_num - 1
            ):
                batch_results = [dct.dct2(b) for b in blocks]
                for (h_b, w_b), result in zip(positions, batch_results):
                    dct_img[
                        h_b * BLOCK_SIZE : (h_b + 1) * BLOCK_SIZE,
                        w_b * BLOCK_SIZE : (w_b + 1) * BLOCK_SIZE,
                    ] = result
                blocks = []
                positions = []

    return dct_img


def calc_V(dct_img, h_block_num, w_block_num):
    V_average = torch.zeros((BLOCK_SIZE, BLOCK_SIZE), device=device)
    for j in range(BLOCK_SIZE):
        for i in range(BLOCK_SIZE):
            w_indices = (
                BLOCK_SIZE
                + torch.arange(1, w_block_num - 2, device=device) * BLOCK_SIZE
                + i
            )
            h_indices = (
                BLOCK_SIZE
                + torch.arange(1, h_block_num - 2, device=device) * BLOCK_SIZE
                + j
            )

            h_idx, w_idx = torch.meshgrid(h_indices, w_indices, indexing="ij")

            a = dct_img[h_idx, w_idx]
            b = dct_img[h_idx, w_idx - BLOCK_SIZE]
            c = dct_img[h_idx, w_idx + BLOCK_SIZE]
            d = dct_img[h_idx - BLOCK_SIZE, w_idx]
            e = dct_img[h_idx + BLOCK_SIZE, w_idx]

            V = torch.sqrt((b + c - 2 * a) ** 2 + (d + e - 2 * a) ** 2)
            V_average[j, i] = torch.mean(V).item()

    return V_average


def process_image_blockiness(img, dct):
    with torch.inference_mode():
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray_img.shape
        cal_height, cal_width, h_margin, w_margin = calc_margin(height, width)
        h_block_num, w_block_num = cal_height // BLOCK_SIZE, cal_width // BLOCK_SIZE

        dct_img = calc_DCT(gray_img, dct, h_block_num, w_block_num)
        dct_cropped_img = calc_DCT(gray_img[4:, 4:], dct, h_block_num, w_block_num)

        V_average = calc_V(dct_img, h_block_num, w_block_num)
        Vc_average = calc_V(dct_cropped_img, h_block_num, w_block_num)
        D = torch.abs((Vc_average - V_average) / (V_average + 1e-8))

        return float(D.sum().cpu())


dct = DCT()

image = cv2.imread("test.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

start = time.time()
blockiness_score = process_image_blockiness(image, dct)
end = time.time()

print("Blockiness: ", blockiness_score)
print("Time: ", end - start)
