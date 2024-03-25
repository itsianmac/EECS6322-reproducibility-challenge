import json
import random
from typing import Dict, List, Tuple


class GQA_Sampler:
    def __init__(self, opts):
        self.opts = {}

        print("Loading GQA questions")
        for opt in opts:
            name = opt["opt_name"]
            self.opts[name] = {}
            self.opts[name]["samples_per_group"] = opt["samples_per_group"]
            self.opts[name]["questions"] = json.load(open(opt["path"]))
            self.opts[name]["data_keys"] = opt["data_keys"]
            self.opts[name]["group_key"] = opt["group_key"]
            self.opts[name]["keys"] = self.opts[name]["questions"].keys()

        print("Loading GQA questions complete!")

    def get_samples(self, sample_type: str):

        # Unroll info from opts dictionary...
        if sample_type in self.opts:
            samples_per_group = self.opts[sample_type]["samples_per_group"]
            questions = self.opts[sample_type]["questions"]
            data_keys = self.opts[sample_type]["data_keys"]
            group_type = self.opts[sample_type]["group_key"]
            keys = self.opts[sample_type]["keys"]
        else:
            raise ValueError(f"Invalid type. Must be one of {self.opts.keys()}")

        # ===== Get testdev GQA samples =====
        # Get unique groups from the test set
        unique_groups = set(data["groups"][group_type] for data in questions.values())

        questions_grouped_by_group = {}

        # Build a dictionary of unique groups and the questions that belong to them
        for key in keys:

            data = questions[key]
            group = data["groups"][group_type]

            question_data = {}
            for relevant_key in data_keys:
                question_data[relevant_key] = data[relevant_key]

            if group not in questions_grouped_by_group:
                questions_grouped_by_group[group] = [question_data]
            else:
                questions_grouped_by_group[group].append(question_data)

        sample = []
        print(f"Unique {sample_type} Groups: ", unique_groups)
        for group in unique_groups:
            # Randomly sample testdev_samples questions from each group
            group_to_sample_from = questions_grouped_by_group[group]
            group_sample = []

            while len(group_sample) < samples_per_group:
                random_key = random.choice(group_to_sample_from)
                group_sample.append(random_key)

            # Ensure that the group has the correct number of samples
            assert len(group_sample) == samples_per_group

            # Unroll the group sample so list is 1D when appending to the testdev sample
            sample.extend(group_sample)

        return sample

    def get_visprog_gqa_samples(self, sample_sets: List) -> List[List[Dict]]:
        """
        Samples based on the options provided in the constructor.

        Parameters
        ----------
        sample_sets : List
            List of strings specifying the sample sets to return.
            Must appear in the opts under the opt_name key.

        Returns
        -------
        List[List[Dict]]
            List of samples, each containing a list of dictionaries.

        """

        print("Initializing VisProg Sampling")

        # ===== Get GQA samples =====
        samples = []
        for sample_type in sample_sets:
            sample = self.get_samples(sample_type)
            samples.append(sample)

        print("Initializing VisProg Sampling complete!")

        return samples


if __name__ == "__main__":
    # Example usage

    # NOTE: way more local groups than global groups... so we'll use global groups
    #       due to chatgpt limitations... even the global groups are probably too much
    """
    To evaluate on a diverse set of question types (~100 detailed types),
    we randomly sample up to k samples per question type from the balanced val
    (k = 5) and test-dev (k = 20) sets."
    """

    group_key = "global"  # one of 'local', 'global',
    validation_opts = {
        "opt_name": "val",
        "path": "../data/GQA/val_balanced_questions.json",
        "data_keys": ["imageId", "question", "answer", "fullAnswer", "groups"],
        "samples_per_group": 5,
        "group_key": group_key,  # one of 'local', 'global'
    }

    test_opts = {
        "opt_name": "testdev",
        "path": "../data/GQA/testdev_balanced_questions.json",
        "data_keys": ["imageId", "question", "answer", "fullAnswer", "groups"],
        "samples_per_group": 20,
        "group_key": group_key,  # one of 'local', 'global'
    }

    opts = [validation_opts, test_opts]
    sample_sets = [opt["opt_name"] for opt in opts]

    gqa_sampler = GQA_Sampler(opts)

    val_samples, testdev_samples = gqa_sampler.get_visprog_gqa_samples(sample_sets)

    # Print out the validation samples
    print(f"Val Sample: {val_samples}")
    print(f"Number of Val Samples: {len(val_samples)}")

    # Print out the testdev samples
    print(f"Testdev Sample: {testdev_samples}")
    print(f"Number of Testdev Samples: {len(testdev_samples)}")
