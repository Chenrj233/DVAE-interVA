import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm, trange



def distinctness_all(generations_df):
    dist1, dist2, dist3 = [], [], []
    unigrams, bigrams, trigrams = set(), set(), set()
    total_words = 0
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i in trange(len(generations_df), desc='Evaluating diversity'):
        generations = str(generations_df[0][i])
        o = generations.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])

    dist1.append(len(unigrams) / total_words)
    dist2.append(len(bigrams) / total_words)
    dist3.append(len(trigrams) / total_words)
    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

def main(generations_file):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    generations_df = pd.read_csv(generations_file, header=None)


    # calculate diversity
    dist1, dist2, dist3 = distinctness_all(generations_df)

    # write output results
    with open(output_dir / 'eval_results.txt', 'w') as fo:
        for i, dist_n in enumerate([dist1, dist2, dist3]):
            fo.write(f'dist-{i + 1} = {dist_n}\n')



if __name__ == '__main__':
    file_name = "samples/amazon.csv"
    main(file_name)
