import torch
from src.masks import create_padding_mask, create_decoder_mask

SOS_IDX = 1
EOS_IDX = 2

def translate(sentence, model, vocab_transform, token_transform,
              src_language, tgt_language, device, max_len=50):
    model.eval()

    tokens     = token_transform[src_language](sentence)
    src_ids    = vocab_transform[src_language](tokens)
    src_tensor = torch.cat([
        torch.tensor([SOS_IDX]),
        torch.tensor(src_ids, dtype=torch.long),
        torch.tensor([EOS_IDX])
    ]).unsqueeze(0).to(device)

    src_mask = create_padding_mask(src_tensor).to(device)

    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)

    tgt_ids = [SOS_IDX]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
        tgt_mask   = create_decoder_mask(tgt_tensor).to(device)

        with torch.no_grad():
            decoder_output = model.decode(tgt_tensor, encoder_output, src_mask, tgt_mask)
            logits         = model.output_linear(decoder_output[:, -1, :])
            next_token     = logits.argmax(dim=-1).item()

        tgt_ids.append(next_token)

        if next_token == EOS_IDX:
            break

    itos   = vocab_transform[tgt_language].get_itos()
    tokens = [itos[i] for i in tgt_ids[1:] if i != EOS_IDX]
    return ' '.join(tokens)
