import argparse
from typing import Dict, Any

from evaluation.common import aggregate_without_voting, aggregate_with_voting, build_figure
from evaluation.nlvr import compute_stats


def process_results(results_file: str) -> Dict[str, Any]:
    stats = compute_stats(results_file)
    without_voting = aggregate_without_voting(stats)
    with_voting = aggregate_with_voting(stats)
    return dict(
        without_voting=without_voting,
        with_voting=with_voting,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Generate figure 6 bar plot for given the NLVR evaluation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'in_context_2_results_file',
        type=str,
    )
    parser.add_argument(
        'in_context_4_results_file',
        type=str,
    )
    parser.add_argument(
        'in_context_8_results_file',
        type=str,
    )
    parser.add_argument(
        'in_context_12_results_file',
        type=str,
    )
    parser.add_argument(
        'figure_file',
        type=str,
    )

    args = parser.parse_args()

    results = {
        '2': process_results(args.in_context_2_results_file),
        '4': process_results(args.in_context_4_results_file),
        '8': process_results(args.in_context_8_results_file),
        '12': process_results(args.in_context_12_results_file),
    }

    fig = build_figure(results)
    fig.savefig(args.figure_file)


if __name__ == '__main__':
    main()
