"""SongFormer sliding-window inference and postprocessing.

Runs the SongFormer model over audio embeddings in TIME_DUR windows,
accumulates logits, and converts to timestamped structure segments.
"""

import math

import numpy as np
import torch
from loguru import logger

from .songformer_embeddings import INPUT_SR, compute_embeddings
from .songformer_setup import SongFormerStack, setup_paths

# Constants from official SongFormer infer.py.
TIME_DUR = 420
AFTER_DS_FR = 8.333
HOP_SIZE = 420
NUM_CLASSES = 128

DATASET_LABEL = "SongForm-HX-8Class"
DATASET_IDS = [5]


def _rule_post_processing(msa_list: list) -> list:
    """Clean up short/duplicate segments at boundaries."""
    if len(msa_list) <= 2:
        return msa_list
    result = msa_list.copy()

    while len(result) > 2:
        if result[1][0] - result[0][0] < 1.0:
            result[0] = (result[0][0], result[1][1])
            result = [result[0]] + result[2:]
        else:
            break
    while len(result) > 2:
        if result[-1][0] - result[-2][0] < 1.0:
            result = result[:-2] + [result[-1]]
        else:
            break
    while len(result) > 2:
        if result[0][1] == result[1][1] and result[1][0] <= 10.0:
            result = [(result[0][0], result[0][1])] + result[2:]
        else:
            break
    while len(result) > 2:
        last_dur = result[-1][0] - result[-2][0]
        if result[-2][1] == result[-3][1] and last_dur <= 10.0:
            result = result[:-2] + [result[-1]]
        else:
            break
    return result


def _build_label_mask(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dataset IDs tensor and label mask for inference."""
    setup_paths()
    from dataset.label2id import (
        DATASET_ID_ALLOWED_LABEL_IDS,
        DATASET_LABEL_TO_DATASET_ID,
    )

    dataset_ids = torch.tensor(DATASET_IDS).to(device, dtype=torch.long)
    ds_id = DATASET_LABEL_TO_DATASET_ID[DATASET_LABEL]
    mask_arr = np.ones(NUM_CLASSES, dtype=bool)
    mask_arr[DATASET_ID_ALLOWED_LABEL_IDS[ds_id]] = False
    label_mask = (
        torch.tensor(mask_arr)
        .to(device, dtype=torch.bool)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    return dataset_ids, label_mask


def run_inference(
    stack: SongFormerStack,
    audio: torch.Tensor,
    device: str = "cuda:0",
) -> list[dict]:
    """Run sliding-window inference and return structure segments.

    Args:
        stack: Loaded SongFormerStack.
        audio: Audio tensor on device, sampled at INPUT_SR.
        device: CUDA device string.

    Returns:
        List of segment dicts with "label", "start", "end" keys.
    """
    setup_paths()
    from postprocessing.functional import postprocess_functional_structure

    total_len = (
        (audio.shape[0] // INPUT_SR // TIME_DUR) * TIME_DUR + TIME_DUR
    )
    total_frames = math.ceil(total_len * AFTER_DS_FR)

    logits = {
        "function_logits": np.zeros([total_frames, NUM_CLASSES]),
        "boundary_logits": np.zeros([total_frames]),
    }
    logits_num = {
        "function_logits": np.zeros([total_frames, NUM_CLASSES]),
        "boundary_logits": np.zeros([total_frames]),
    }

    dataset_ids, label_mask = _build_label_mask(device)

    lens = 0
    i = 0
    with torch.no_grad():
        while True:
            if i * INPUT_SR >= audio.shape[-1]:
                break
            end_idx = min((i + HOP_SIZE) * INPUT_SR, audio.shape[-1])
            if end_idx - i * INPUT_SR <= 1024:
                break

            embd = compute_embeddings(
                audio, stack.muq, stack.musicfm, i, HOP_SIZE,
            )
            _, chunk_logits = stack.model.infer(
                input_embeddings=embd,
                dataset_ids=dataset_ids,
                label_id_masks=label_mask,
                with_logits=True,
            )

            sf = int(i * AFTER_DS_FR)
            ef = sf + min(
                math.ceil(HOP_SIZE * AFTER_DS_FR),
                chunk_logits["boundary_logits"][0].shape[0],
            )
            logits["function_logits"][sf:ef] += (
                chunk_logits["function_logits"][0].cpu().numpy()
            )
            logits["boundary_logits"][sf:ef] = (
                chunk_logits["boundary_logits"][0].cpu().numpy()
            )
            logits_num["function_logits"][sf:ef] += 1
            logits_num["boundary_logits"][sf:ef] += 1
            lens += ef - sf
            i += HOP_SIZE

    logits["function_logits"] /= logits_num["function_logits"]
    logits["boundary_logits"] /= logits_num["boundary_logits"]
    logits["function_logits"] = torch.from_numpy(
        logits["function_logits"][:lens],
    ).unsqueeze(0)
    logits["boundary_logits"] = torch.from_numpy(
        logits["boundary_logits"][:lens],
    ).unsqueeze(0)

    msa_output = postprocess_functional_structure(logits, stack.hp)
    assert msa_output[-1][-1] == "end"
    msa_output = _rule_post_processing(msa_output)

    segments = []
    for idx in range(len(msa_output) - 1):
        segments.append({
            "label": msa_output[idx][1],
            "start": msa_output[idx][0],
            "end": msa_output[idx + 1][0],
        })
    return segments
