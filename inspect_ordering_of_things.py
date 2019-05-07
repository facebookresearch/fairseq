import pickle 
import numpy as np


with open('ordered_prompt_ranking_scores.pkl', 'rb') as handle:
    ordered_prompt_ranking_scores = pickle.load(handle)
with open('ordered_prompts_w_indices.pkl', 'rb') as handle2:
    ordered_prompts_w_indices = pickle.load(handle2)
with open('ordered_sents_w_indices.pkl', 'rb') as handle3:
    ordered_sents_w_indices = pickle.load(handle3)

just_probs = [x[1] for x in ordered_prompt_ranking_scores]

for i in range(0, len(ordered_prompt_ranking_scores), 10):
    curr_scores = ordered_prompt_ranking_scores[i: i+10]
    curr_just_probs = just_probs[i: i+10]
    curr_prompts = ordered_prompts_w_indices[i: i+10]
    curr_stories = ordered_sents_w_indices[i:i+10]

    max_val = np.argmax(curr_just_probs)

    print('i: {}'.format(i))
    print('Max Val: {}'.format(max_val))
    print('Curr just probs: {}'.format(curr_just_probs))
    print('Curr Scores: {}'.format(curr_scores))
    print('Curr Prompts: {}'.format(curr_prompts))
    print('Curr Stories: {}'.format(curr_stories))


