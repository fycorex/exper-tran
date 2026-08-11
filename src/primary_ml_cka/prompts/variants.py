from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT

COMPACT_CLASSIFICATION_PROMPT = """\
Identify the main vehicle. Choose exactly one option:
1 ambulance, 2 cab, 3 limousine, 4 minivan, 5 sports car,
6 fire engine, 7 garbage truck, 8 pickup truck, 9 tow truck, 10 moving van.
Answer with only the option number."""

QUESTION_CLASSIFICATION_PROMPT = """\
What is the main object in this image?
Options: (1) ambulance; (2) cab; (3) limousine; (4) minivan;
(5) sports car; (6) fire engine; (7) garbage truck; (8) pickup truck;
(9) tow truck; (10) moving van.
Output exactly one number from 1 through 10 and nothing else."""

PROMPT_VARIANTS = {
    "original": CLASSIFICATION_PROMPT,
    "compact": COMPACT_CLASSIFICATION_PROMPT,
    "question": QUESTION_CLASSIFICATION_PROMPT,
}


def get_prompt(prompt_id: str) -> str:
    try:
        return PROMPT_VARIANTS[prompt_id]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt ID: {prompt_id}") from exc
