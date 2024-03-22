import argparse
import json
import os

import yaml
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Extract prompts YAML file from raw NLVR dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'nlvr_metadata',
        type=str,
        help='Path to the NLVR metadata file (e.g., test1.json)',
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the directory containing the NLVR images',
    )
    parser.add_argument(
        'prompt_file',
        type=str,
        help='Path to the output YAML file',
    )

    args = parser.parse_args()

    with open(args.nlvr_metadata, 'r') as f:
        raw_data = [json.loads(line) for line in f]

    prompts_by_sentence = {}
    extracted_pairs = 0
    for sample in tqdm(raw_data, total=len(raw_data)):
        split, set_id, pair_id, sentence_id = sample['identifier'].split('-')
        sentence_uid = f'{split}-{set_id}-{sentence_id}'
        if sentence_uid not in prompts_by_sentence:
            prompts_by_sentence[sentence_uid] = dict(
                sentence=sample['sentence'],
                pairs=[],
            )
        image_prefix = '-'.join(sample['identifier'].split('-')[:-1])
        left_image = f'{image_prefix}-img0.png'
        right_image = f'{image_prefix}-img1.png'
        if not os.path.exists(os.path.join(args.data_dir, left_image)):
            continue
        if not os.path.exists(os.path.join(args.data_dir, right_image)):
            continue
        assert prompts_by_sentence[sentence_uid]['sentence'] == sample['sentence'], \
            f'Sentence mismatch for {sentence_uid} for pair {pair_id}'
        extracted_pairs += 1
        prompts_by_sentence[sentence_uid]['pairs'].append(dict(
            id=int(pair_id),
            left_image=left_image,
            right_image=right_image,
            label=sample['label'].lower() == 'true',
        ))

    prompts = []
    for sentence_uid, sentence_data in prompts_by_sentence.items():
        prompts.append(dict(
            id=sentence_uid,
            prompt=dict(
                statement=sentence_data['sentence'],
            ),
            pairs=sentence_data['pairs'],
        ))

    print(f'Extracted {len(prompts)} sentences with a total of {extracted_pairs} pairs')

    os.makedirs(os.path.dirname(args.prompt_file), exist_ok=True)
    with open(args.prompt_file, 'w') as f:
        yaml.dump(prompts, f, sort_keys=False)


if __name__ == '__main__':
    main()
