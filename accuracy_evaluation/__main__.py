"""Entry point for accuracy_evaluation module when run as: python -m accuracy_evaluation"""

import sys

if __name__ == "__main__":
    from accuracy_evaluation.cli import main

    sys.exit(main())
