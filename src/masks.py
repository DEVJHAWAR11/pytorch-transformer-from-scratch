import torch


def create_padding_mask(seq, pad_idx=0):
    """
    seq: (batch, seq_len) — token ids
    returns: (batch, 1, 1, seq_len) float mask
    1 = attend, 0 = ignore (padding)
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.float()


def create_look_ahead_mask(seq_len):
    """
    returns: (seq_len, seq_len) lower triangular mask
    1 = attend, 0 = blocked (future token)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


def create_decoder_mask(tgt_seq, pad_idx=0):
    """
    Combines padding mask and look-ahead mask for decoder self-attention.
    tgt_seq: (batch, tgt_len)
    returns: (batch, 1, tgt_len, tgt_len)
    """
    tgt_len = tgt_seq.size(1)
    tgt_padding_mask    = create_padding_mask(tgt_seq, pad_idx)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt_seq.device)
    combined_mask = torch.minimum(tgt_padding_mask, tgt_look_ahead_mask)
    return combined_mask
