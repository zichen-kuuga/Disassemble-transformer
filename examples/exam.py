import torch
from transformer.model import Transformer
from transformer.mask import create_padding_mask

model=Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    num_layers=6,
)

src=torch.randint(0,10000,(32,10))
tgt=torch.randint(0,10000,(32,12))

src_mask, tgt_mask = create_padding_mask(src,tgt)

logits=model(src,tgt,src_mask,tgt_mask)
print(logits.shape)