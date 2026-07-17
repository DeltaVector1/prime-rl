from collections import defaultdict
from string import ascii_uppercase

import verifiers as vf
from datasets import Dataset, DatasetDict, load_dataset
from verifiers.utils.data_utils import extract_boxed_answer


FINAL_ANSWER_INSTRUCTION = (
    "Reason through the question, then put only the letter of the correct answer in \\boxed{}."
)


def _format_question(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{ascii_uppercase[index]}. {choice}" for index, choice in enumerate(choices))
    return f"{question}\n{options}"


def _format_few_shot(row: dict) -> str:
    answer = ascii_uppercase[int(row["answer"])]
    return f"{_format_question(row['question'], row['choices'])}\nAnswer: {answer}"


def load_environment(
    dataset_name: str = "cais/mmlu",
    dataset_subset: str = "all",
    dataset_split: str = "test",
    num_few_shot: int = 5,
    shuffle_seed: int = 42,
    system_prompt: str | None = None,
    **_kwargs,
) -> vf.Environment:
    def build_eval_dataset() -> Dataset:
        dataset = load_dataset(dataset_name, dataset_subset)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected DatasetDict from {dataset_name}, got {type(dataset).__name__}")
        if dataset_split not in dataset:
            raise ValueError(f"Dataset split {dataset_split!r} is unavailable; found {sorted(dataset)}")

        dev_by_subject: dict[str, list[dict]] = defaultdict(list)
        for row in dataset["dev"]:
            dev_by_subject[str(row["subject"])].append(row)

        rows = []
        for index, row in enumerate(dataset[dataset_split]):
            subject = str(row["subject"])
            few_shot = dev_by_subject[subject][:num_few_shot]
            context = "\n\n".join(_format_few_shot(example) for example in few_shot)
            question = _format_question(row["question"], row["choices"])
            prompt_parts = [f"The following are multiple-choice questions about {subject.replace('_', ' ')}."]
            if context:
                prompt_parts.append(context)
            prompt_parts.extend((question, FINAL_ANSWER_INSTRUCTION))
            answer = ascii_uppercase[int(row["answer"])]
            rows.append(
                {
                    "question": "\n\n".join(prompt_parts),
                    "answer": answer,
                    "info": {"id": f"{subject}:{index}", "subject": subject},
                }
            )

        return Dataset.from_list(rows).shuffle(seed=shuffle_seed)

    parser = vf.MaybeThinkParser(extract_boxed_answer)
    rubric = vf.MathRubric(parser=parser)
    return vf.SingleTurnEnv(
        eval_dataset=build_eval_dataset,
        parser=parser,
        rubric=rubric,
        system_prompt=system_prompt,
    )
