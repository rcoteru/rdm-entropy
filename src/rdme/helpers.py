from __future__ import annotations

from collections.abc import Sequence

import torch

""" Small analysis utilities, used by the plotting stages of the scripts rather than by the
model. One function so far; the file exists so that an array utility does not have to live in
a physics module to be shared between scripts. """


def grab_closest_idxs(vector: torch.Tensor, values: Sequence[float]) -> list[int]:
    """ Indices of the entries of `vector` closest to each of `values`. """
    return [int(torch.argmin((vector - value).abs())) for value in values]
