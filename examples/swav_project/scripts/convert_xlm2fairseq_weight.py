import os
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlm_model', type=str, help="Input XLM model")
    parser.add_argument('--fairseq_model', type=str, help="Input fairseq model")
    parser.add_argument('--output', type=str, help="Input XLM model")
    parser.add_argument('--seq2seq', action='store_true', default=False, help="If the model is seq2seq")

    args = parser.parse_args()
    assert os.path.exists(args.xlm_model), f'{args.xlm_model} not found.'
    assert os.path.exists(args.fairseq_model), f'{args.fairseq_model} not found.'

    xlm_m = torch.load(args.xlm_model, map_location='cpu')
    fairseq_m = torch.load(args.fairseq_model, map_location='cpu')
    if args.seq2seq:
        # xlm: ['encoder': modules, 'decoder': modules]
        # fairseq: ['model': modules{'encoder.'..., 'decoder.'...}]
        # NOTE: check if xlm has module.
        assert 'encoder' in xlm_m
        assert 'decoder' in xlm_m
        xlm_enc_map = {'encoder.' + k.replace('module.', ''): v for k, v in xlm_m['encoder'].items()}
        xlm_dec_map = {'decoder.' + k.replace('module.', ''): v for k, v in xlm_m['decoder'].items()}
        # assert all(k in fairseq_m['model'] for k in xlm_enc_map.keys())
        # assert all(k in fairseq_m['model'] for k in xlm_dec_map.keys())
        for d in [xlm_enc_map, xlm_dec_map]:
            for k, v in d.items():
                if k not in fairseq_m['model']:
                    print(f'WARNING: {k} not in fairseq_m')
                # assert k in fairseq_m['model'], f'{k} not found in fairseq_m, {fairseq_m["model"].keys()}'
                fairseq_m["model"][k] = v
    else:
        # xlm: ['model': modules]
        # fairseq: ['model': modules{'encoder.'...}]
        assert 'model' in xlm_m or 'encoder' in xlm_m
        xlm_key = 'model' if 'model' in xlm_m else 'encoder'
        xlm_map = {'encoder.' + k.replace('module.', ''): v for k, v in xlm_m[xlm_key].items()}
        for k, v in xlm_map.items():
            assert k in fairseq_m['model'], f'{k} not found in fairseq_m, {fairseq_m["model"].keys()}'
            fairseq_m["model"][k] = v
    torch.save(fairseq_m, args.output)




