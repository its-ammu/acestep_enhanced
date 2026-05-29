"""SongFormer audio embedding computation.

Computes fused MuQ + MusicFM embeddings for sliding-window inference.
Called by songformer_inference.py.
"""

import torch

# Constants from official SongFormer infer.py.
INPUT_SR = 24000
WIN_SIZE = 420


def compute_embeddings(
    audio: torch.Tensor,
    muq,
    musicfm,
    start_sec: int,
    hop_sec: int,
) -> torch.Tensor:
    """Compute fused MuQ+MusicFM embeddings for one audio window.

    Args:
        audio: Full audio tensor on device, shape (samples,).
        muq: Loaded MuQ model.
        musicfm: Loaded MusicFM model.
        start_sec: Window start in seconds.
        hop_sec: Window hop size in seconds.

    Returns:
        Concatenated embedding tensor, shape (1, frames, 4096).
    """
    start_idx = start_sec * INPUT_SR
    end_idx = min((start_sec + WIN_SIZE) * INPUT_SR, audio.shape[-1])
    audio_seg = audio[start_idx:end_idx]

    muq_out = muq(audio_seg.unsqueeze(0), output_hidden_states=True)
    muq_420 = muq_out["hidden_states"][10]
    del muq_out
    torch.cuda.empty_cache()

    _, fm_hidden = musicfm.get_predictions(audio_seg.unsqueeze(0))
    fm_420 = fm_hidden[10]
    del fm_hidden
    torch.cuda.empty_cache()

    muq_chunks, fm_chunks = [], []
    for idx_30 in range(start_sec, start_sec + hop_sec, 30):
        s = idx_30 * INPUT_SR
        e = min(
            (idx_30 + 30) * INPUT_SR,
            audio.shape[-1],
            (start_sec + hop_sec) * INPUT_SR,
        )
        if s >= audio.shape[-1] or e - s <= 1024:
            break
        seg = audio[s:e].unsqueeze(0)
        muq_chunks.append(
            muq(seg, output_hidden_states=True)["hidden_states"][10],
        )
        torch.cuda.empty_cache()
        fm_chunks.append(musicfm.get_predictions(seg)[1][10])
        torch.cuda.empty_cache()

    muq_30 = torch.cat(muq_chunks, dim=1)
    fm_30 = torch.cat(fm_chunks, dim=1)

    all_embds = [fm_30, muq_30, fm_420, muq_420]
    min_len = min(e.shape[1] for e in all_embds)
    all_embds = [e[:, :min_len, :] for e in all_embds]
    return torch.cat(all_embds, dim=-1)
