from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--txt", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--gpu", type=int, default=1)
args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
    base_model.resize_token_embeddings(260164)
    tokenizer = AutoTokenizer.from_pretrained('MaLA-LM/mala-500')
    if args.gpu == 1:
        model = PeftModel.from_pretrained(base_model, 'MaLA-LM/mala-500').to("cuda")
    else:
        model = PeftModel.from_pretrained(base_model, 'MaLA-LM/mala-500')
    model.eval()

    txts = [x.strip() for x in open(args.txt, "r").readlines()]

    with open(args.dst + "/lm_score", "w", buffering=1) as f:
        for t in tqdm(txts):
            input_tokens = tokenizer("", add_special_tokens=True, return_tensors='pt').input_ids
            if len(t) > 0:
                output_tokens = tokenizer(t, add_special_tokens=False, return_tensors='pt').input_ids
                tokens = torch.cat([input_tokens, output_tokens], dim=1)
                length = output_tokens.shape[-1]
            else:
                tokens = input_tokens
                length = 0

            if args.gpu == 1:
                tokens = tokens.to("cuda")

            with torch.no_grad():
                outputs = model(tokens)
                logits = outputs.logits
            
            log_sum = 0
            for i in range(tokens.shape[-1] - 1):
                past_tok, current_tok = i, i + 1
                token_logit = logits[0, past_tok, :]
                token_log_probs = torch.nn.functional.log_softmax(token_logit, dim=-1)
                log_token_prob = token_log_probs[tokens[0, current_tok]].item()
                log_sum += log_token_prob

            f.write(str(log_sum) + "\n")