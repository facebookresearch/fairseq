import pickle
import numpy as np
from tqdm import tqdm
import argparse

NUM_STORIES = 1000
NUM_FAKE_PROMPTS = 9

"""
To run:

python make_prompt_ranking.py --datapath examples/stories/writingPrompts/test

"""


def main(args):
    datapath = args.datapath
    src_in = datapath + ".wp_source"
    tgt_in = datapath + ".wp_target"
    print('Loading source text from: {} and target text from: {}'.format(src_in, tgt_in))

    with open(src_in, 'r') as g:
        prompts = g.readlines()
    with open(tgt_in, 'r') as f:
        stories = f.readlines()

    # Sample 1000 story indices
    random_story_idxs = np.random.choice(np.arange(len(stories)), NUM_STORIES, replace=False)

    # Start writing to file. We frame this as a 'translation' task:
    # src_out will have 1000*10 prompts, in groups of 10, always with the correct prompt first
    # tgt_out will have 1000 stories, each duplicated 10 times

    src_out = datapath + "_promptranking.wp_source"
    tgt_out = datapath + "_promptranking.wp_target"

    print("Writing prompt ranking text. wp_source to: {} and wp_target out to: {}".format(src_out, tgt_out))

    with open(src_out, 'w') as wpsource:
        with open(tgt_out, 'w') as wptarget:

            for story_idx in tqdm(random_story_idxs):
                story = stories[story_idx]
                correct_prompt = prompts[story_idx]

                # Sample 9 other prompts from rest of prompts, excluding the currently sampled gold prompt-story pair
                rest_of_indices = list(np.arange(len(stories)))[:story_idx] + list(np.arange(len(stories)))[story_idx + 1:]
                fake_prompt_idxs = np.random.choice(rest_of_indices, NUM_FAKE_PROMPTS, replace=False)
                fake_prompts = list(np.array(prompts)[fake_prompt_idxs])

                # Write original (prompt,story) pair
                wpsource.write(correct_prompt.strip() + "\n")
                wptarget.write(story.strip() + "\n")

                # Write the (fake prompt, story) pairs
                for fake_prompt in fake_prompts:
                    wpsource.write(fake_prompt.strip() + "\n")
                    wptarget.write(story.strip() + "\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True, help='Path/prefix for the .wp_source and .wp_target files to format into promptranking text')
    args = parser.parse_args()

    np.random.seed(0)  # set random seed

    main(args)
