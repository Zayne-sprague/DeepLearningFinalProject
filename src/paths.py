from pathlib import Path

__CURR_DIR__ = Path(__file__).resolve().parent

GOOGLE_COLAB = False


ROOT_PROJECT_DIR = (__CURR_DIR__ / '../').absolute().resolve()

SRC_DIR = ROOT_PROJECT_DIR / 'src'
CHECKPOINTS_DIR = SRC_DIR / 'checkpoints'
DATA_DIR = SRC_DIR / "data"
SMALL_DATA_DIR = SRC_DIR / "small_data"
CONFIGS_DIR = SRC_DIR / "configs"

__GOOGLE_COLAB_STATS__PATH = Path('/content/gdrive/MyDrive/colab_output/deep_learning_final_project/stats')
__GOOGLE_COLAB_IMG__PATH = Path('/content/gdrive/MyDrive/colab_output/deep_learning_final_project/img')

STATS_DIR = SRC_DIR / "stats"
IMAGES_DIR = ROOT_PROJECT_DIR / 'images'

if GOOGLE_COLAB:
    STATS_DIR = __GOOGLE_COLAB_STATS__PATH
    IMAGES_DIR = __GOOGLE_COLAB_IMG__PATH


CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
