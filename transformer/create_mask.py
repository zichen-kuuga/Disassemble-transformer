def create_padding_mask(src, tgt, pad_idx=0):
    # src: (batch_size, src_seq_len)
    # tgt: (batch_size, tgt_seq_len)

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)    # (batch_size, 1, 1, src_seq_len)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)    # (batch_size, 1, 1, tgt_seq_len)

    tgt_len =tgt.size(1)
    look_ahead_mask = torch.ones(tgt_len, tgt_len).tril().bool().unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_mask & look_ahead_mask.to(tgt.device)    # (batch_size, 1, tgt_seq_len, tgt_seq_len)

    return src_mask, tgt_mask