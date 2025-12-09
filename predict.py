import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.pipeline import InferencePipeline
from src.postprocessing import SubmissionValidator


def main():

    print("MABe Mouse Behavior Detection - Inference")
    print("=" * 60)

    pipeline = InferencePipeline(Config)

    output_path = Config.SUBMISSION_DIR / "submission.csv"
    submission = pipeline.run(output_path=str(output_path))

    print("\nValidating submission...")
    SubmissionValidator.validate(submission)

    print("\nInference complete!")
    print(f"Submission saved to: {output_path}")


if __name__ == "__main__":
    main()
