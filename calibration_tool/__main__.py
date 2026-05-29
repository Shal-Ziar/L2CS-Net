"""Entry point for calibration module when run as: python -m calibration"""

import sys

if __name__ == "__main__":
    from calibration_tool.cli import main
    sys.exit(main())
