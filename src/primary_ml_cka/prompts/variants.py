from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT

COMPACT_CLASSIFICATION_PROMPT = """\
Identify the main vehicle. Choose exactly one option:
0 ambulance, 1 cab, 2 limousine, 3 minivan, 4 sports car,
5 fire engine, 6 garbage truck, 7 pickup truck, 8 tow truck, 9 moving van.
Answer with only the option number."""

QUESTION_CLASSIFICATION_PROMPT = """\
What is the main object in this image?
Options: (0) ambulance; (1) cab; (2) limousine; (3) minivan;
(4) sports car; (5) fire engine; (6) garbage truck; (7) pickup truck;
(8) tow truck; (9) moving van.
Output exactly one number from 0 through 9 and nothing else."""

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
