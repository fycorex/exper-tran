from collections.abc import Callable

from primary_ml_cka.attack.losses.primary import PrimaryLoss

AttackCallback = Callable[[int, PrimaryLoss, object], None]
