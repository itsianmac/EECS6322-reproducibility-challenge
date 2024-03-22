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

    prompts = []
    for sample in tqdm(raw_data, total=len(raw_data)):
        image_prefix = '-'.join(sample['identifier'].split('-')[:-1])
        left_image = f'{image_prefix}-img0.png'
        right_image = f'{image_prefix}-img1.png'
        if not os.path.exists(os.path.join(args.data_dir, left_image)):
            continue
        if not os.path.exists(os.path.join(args.data_dir, right_image)):
            continue
        prompts.append(dict(
            id=sample['identifier'],
            left_image=left_image,
            right_image=right_image,
            prompt=dict(
                statement=sample['sentence'],
            ),
            label=sample['label'].lower() == 'true',
        ))

    os.makedirs(os.path.dirname(args.prompt_file), exist_ok=True)
    with open(args.prompt_file, 'w') as f:
        yaml.dump(prompts, f)


if __name__ == '__main__':
    main()
