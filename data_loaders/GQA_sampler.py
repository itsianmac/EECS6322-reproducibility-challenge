import json
import random

import pudb


class GQA_Sampler:
    def __init__(self, gqa_testdev_questions_path: str, gqa_val_questions_path: str):
        print("Loading GQA questions")
        self.val_questions = json.load(open(gqa_val_questions_path))
        self.val_keys = self.val_questions.keys()
        # The data we care about...
        self.val_data_keys = {"question", "imageId"}

        self.testdev_questions = json.load(open(gqa_testdev_questions_path))
        self.tesdtev_keys = self.testdev_questions.keys()

        # The data we care about...
        self.testdev_data_keys = {
            "question",
            "imageId",
            "groups",
            "answer",
            "fullAnswer",
        }

        print("Loading GQA questions complete!")

    def initialize_visprog_sampling(
        self, val_samples: int = 5, testdev_samples: int = 20
    ):
        print("Initializing VisProg Sampling")
        # Randomly sample 5 questions from the validation set
        self.val_sample = []

        val_sample_chance = val_samples / len(self.val_keys)
        testdev_sample_chance = testdev_samples / len(self.tesdtev_keys)

        for key in self.val_keys:
            if len(self.val_sample) == 5:
                break

            # Val has no groups, so randomly samply 5 questions from the dictionary
            random_val = random.random()
            if random_val < val_sample_chance:
                self.val_sample.append(self.val_questions[key])

        print("Initializing VisProg Sampling complete!")


if __name__ == "__main__":
    # Example usage
    gqa_sampler = GQA_Sampler(
        "../data/GQA/train_balanced_questions.json",
        "../data/GQA/test_balanced_questions.json",
    )
    gqa_sampler.initialize_visprog_sampling()

    # Get unique local groups
    unique_local_groups = set()
    for local_group in gqa_sampler.local_groups:
        unique_local_groups.add(local_group)

    print(f"Unique Local Groups: {unique_local_groups}")
    print(f"Number of Unique Local Groups: {len(unique_local_groups)}")

    # Get unique global groups
    unique_global_groups = set()
    for global_group in gqa_sampler.global_groups:
        unique_global_groups.add(global_group)

    print(f"Unique Global Groups: {unique_global_groups}")
    print(f"Number of Unique Global Groups: {len(unique_global_groups)}")
