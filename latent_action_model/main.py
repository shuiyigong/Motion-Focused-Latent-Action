from lightning.pytorch.cli import LightningCLI
from genie.dataset import LightningOpenX
from genie.model import MFLAM
cli = LightningCLI(
    MFLAM,
    LightningOpenX,
    seed_everything_default=42,
)
