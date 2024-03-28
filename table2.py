import argparse

from evaluation.nlvr import compute_stats, compute_one_run_accuracy
from evaluation.common import aggregate_with_voting


def main():
    parser = argparse.ArgumentParser(
        description='Generate Table 2 for given the NLVR evaluation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        'in_context_all_results_file',
        type=str,
    )

    args = parser.parse_args()

    one_run_accuracy = compute_one_run_accuracy(args.in_context_all_results_file, seed=args.seed)
    voting_accuracy = aggregate_with_voting(compute_stats(args.in_context_all_results_file))

    print(f'One run accuracy: {one_run_accuracy:.2f}')
    print(f'Voting accuracy: {voting_accuracy:.2f}')


if __name__ == '__main__':
    main()
