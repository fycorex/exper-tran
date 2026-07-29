CLASS_NAMES = (
    "ambulance",
    "cab",
    "limousine",
    "minivan",
    "sports car",
    "fire engine",
    "garbage truck",
    "pickup truck",
    "tow truck",
    "moving van",
)

CLASS_SYNSETS = (
    "n02701002",
    "n02930766",
    "n03670208",
    "n03770679",
    "n04285008",
    "n03345487",
    "n03417042",
    "n03930630",
    "n04461696",
    "n03796401",
)


def human_label_to_index(label: int) -> int:
    if not 1 <= label <= 10:
        raise ValueError(label)
    return label - 1
