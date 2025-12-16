import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.pipeline import TrainingPipeline


def main():
    print("MABe Mouse Behavior Detection - Training")
    print("=" * 60)

    pipeline = TrainingPipeline(Config)

    f1_scores = pipeline.run()

    print(f"F1 scores: {f1_scores}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
