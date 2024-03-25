import json
import random
from typing import Dict, List, Tuple


class GQA_Sampler:
    def __init__(self, gqa_val_info: dict, gqa_test_info: dict):

        print("Loading GQA questions")

        gqa_val_path = gqa_val_info["path"]
        gqa_val_data_keys = gqa_val_info["data_keys"]
        self.gqa_val_samples_per_group = gqa_val_info["samples_per_group"]

        self.val_questions = json.load(open(gqa_val_path))

        # Caching all keys for easy access
        self.val_keys = self.val_questions.keys()

        # The data we care about...
        self.val_data_keys = gqa_val_data_keys

        self.val_group = gqa_val_info["group_key"]

        gqa_test_path = gqa_test_info["path"]
        gqa_test_data_keys = gqa_test_info["data_keys"]
        self.gqa_test_samples_per_group = gqa_test_info["samples_per_group"]

        self.testdev_questions = json.load(open(gqa_test_path))

        # Caching all keys for easy access
        self.tesdtev_keys = self.testdev_questions.keys()

        # The data we care about...
        self.testdev_data_keys = gqa_test_data_keys

        self.testdev_group = gqa_test_info["group_key"]

        print("Loading GQA questions complete!")

    def get_samples(self, sample_type: str = "val"):

        # HACK: easier to just make this a dictionary off the top of initialization,
        # or better yet, pass the number of samples etc...
        if sample_type == "val":
            group_type = self.val_group
            questions = self.val_questions
            keys = self.val_keys
            data_keys = self.val_data_keys
            samples_per_group = self.gqa_val_samples_per_group
        elif sample_type == "testdev":
            group_type = self.testdev_group
            questions = self.testdev_questions
            keys = self.tesdtev_keys
            data_keys = self.testdev_data_keys
            samples_per_group = self.gqa_test_samples_per_group
        else:
            raise ValueError("Invalid type. Must be 'val' or 'testdev'")

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
        To evaluate on a diverse set of question types (~100 detailed types),
        we randomly sample up to k samples per question type from the balanced val
        (k = 5) and test-dev (k = 20) sets."


        Parameters
        ----------
        sample_sets : List
            List of strings specifying the sample sets to return. Must be one of 'val' or 'testdev'.

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
    group_key = "global"  # one of 'local', 'global',
    validation_opts = {
        "path": "../data/GQA/val_balanced_questions.json",
        "data_keys": ["imageId", "question", "answer", "fullAnswer", "groups"],
        "samples_per_group": 5,
        "group_key": group_key,  # one of 'local', 'global'
    }

    test_opts = {
        "path": "../data/GQA/testdev_balanced_questions.json",
        "data_keys": ["imageId", "question", "answer", "fullAnswer", "groups"],
        "samples_per_group": 20,
        "group_key": group_key,  # one of 'local', 'global'
    }

    gqa_sampler = GQA_Sampler(validation_opts, test_opts)

    sample_sets = ["val", "testdev"]
    val_samples, testdev_samples = gqa_sampler.get_visprog_gqa_samples(sample_sets)

    # Print out the validation samples
    print(f"Val Sample: {val_samples}")
    print(f"Number of Val Samples: {len(val_samples)}")

    # Print out the testdev samples
    print(f"Testdev Sample: {testdev_samples}")
    print(f"Number of Testdev Samples: {len(testdev_samples)}")
