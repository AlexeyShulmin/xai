from __future__ import annotations

"""Memory‑efficient RISE that works even when you can only fit < 250 masks.

Main idea — **stream the masks in mini‑batches** instead of sending all *N*
masked images through the network at once.

Parameters
----------
model           : Torch network (eval‑mode).
input_size      : (H, W) – size expected by the network.
N               : total number of random masks.
s               : coarse grid size (*s* × *s* masks up‑sampled to H × W).
p               : probability of 1 in the coarse mask.
batch           : **number of masks processed in one forward‑pass**;  tunes
                  memory use.  If not set, falls back to N (old behaviour).
"""

from typing import Tuple
import torch
import torch.nn.functional as F
from math import ceil
from tqdm import tqdm
import matplotlib.pyplot as plt


class RISE:
    def __init__(
        self,
        model: torch.nn.Module,
        input_size: Tuple[int, int],
        *,
        N: int = 8000,
        s: int = 7,
        p: float = 0.5,
        batch: int | None = None,
        device: str | torch.device = "cpu",
        smooth_sigma: float | None = None,
    ) -> None:
        self.model = model.eval()
        self.H, self.W = input_size
        self.N = N
        self.s = s
        self.p = p
        self.batch = batch or N
        self.device = device
        self.smooth_sigma = smooth_sigma
        self._upsample = torch.nn.Upsample((self.H, self.W), mode="bilinear")

    # --------------------------------------------------- mask generator ----
    def _make_coarse(self, n: int) -> torch.Tensor:
        return torch.bernoulli(torch.full((n, 1, self.s, self.s), self.p))

    # ---------------------------------------------------------- explain ----
    @torch.no_grad()
    def explain(self, x: torch.Tensor, *, target: int | None = None) -> torch.Tensor:
        assert x.ndim == 4 and x.size(0) == 1, "x must be shape (1,C,H,W)"
        device = self.device
        x = x.to(device)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        print(torch.round(probs, decimals=3))

        accum: torch.Tensor | None = None
        total_masks = 0

        # stream masks in chunks of size `self.batch`
        for start in tqdm(range(0, self.N, self.batch), total=len(list(range(0, self.N, self.batch)))):
            bs = min(self.batch, self.N - start)
            coarse = self._make_coarse(bs)
            masks = self._upsample(coarse).to(device)  # (bs,1,H,W)
            masked_imgs = x * masks              # broadcast multiply

            # plt.imshow(masked_imgs[0].permute(1, 2, 0).cpu().detach().numpy())
            # plt.show()

            logits = self.model(masked_imgs)
            probs = F.softmax(logits, dim=1)     # (bs, C)
            
            check_list = [probs[i].argmax().item() for i in range(len(probs))]
            check = set(check_list)
            print()
            print()
            print(check)
            print([check_list.count(el) for el in check])
            print()
            if target is None:
                if accum is None:
                    # pick target from first chunk if not supplied
                    target = probs.mean(0).argmax().item()
            weights = probs[:, target].view(bs, 1, 1, 1)
            contrib = (weights * masks).sum(0)   # (1,H,W)

            accum = contrib if accum is None else accum + contrib
            total_masks += bs

        saliency = accum / total_masks           # average
        saliency = F.relu(saliency)
        saliency = saliency / saliency.max()
        saliency = saliency.squeeze(0).cpu()     # (H,W)

        if self.smooth_sigma:
            import cv2, numpy as np
            k = int(ceil(self.smooth_sigma * 3) * 2 + 1)
            saliency_np = cv2.GaussianBlur(saliency.numpy(), (k, k), self.smooth_sigma)
            saliency = torch.from_numpy(saliency_np)
        return saliency
