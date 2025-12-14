"""Entry point for running as a module or via console script."""

import argparse

import kaktus_promo


def main() -> None:
    """Parse arguments and run the promo monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor mujkaktus.cz for new credit doubling promotions"
    )
    parser.add_argument(
        "recipients",
        help="Email address(es) to notify, comma-separated",
    )
    parser.add_argument(
        "-s",
        "--state-file",
        default=kaktus_promo.DEFAULT_STATE_FILE,
        help=f"Path to state file (default: {kaktus_promo.DEFAULT_STATE_FILE})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level) - useful for debugging",
    )

    args = parser.parse_args()
    recipients = [r.strip() for r in args.recipients.split(",")]
    kaktus_promo.main(
        recipients=recipients,
        state_file=args.state_file,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
