# Kaktus Promo Monitor

A Python script that monitors [mujkaktus.cz](https://www.mujkaktus.cz/chces-pridat) for credit-doubling promotional offers and sends email notifications when new promotions are detected.

## What it does

Kaktus (a Czech mobile operator) occasionally runs promotional campaigns where customers can double their prepaid credit. This script:

1. Scrapes the promo page for the Terms & Conditions PDF link
2. Compares it against the previously seen URL
3. If changed, downloads the PDF and extracts the promotion time window
4. Sends an email notification with the promo details
5. Persists state to avoid duplicate notifications

## Requirements

- Python 3.12+
- `sendmail` available at `/usr/sbin/sendmail` for email notifications

## Installation

```bash
make setup
```

## Usage

```bash
# Run silently (suitable for cron)
venv/bin/kaktus-promo user@example.com

# Multiple recipients (comma-separated)
venv/bin/kaktus-promo user1@example.com,user2@example.com

# Custom state file location
venv/bin/kaktus-promo user@example.com -s /path/to/state.json

# Run with verbose logging
venv/bin/kaktus-promo user@example.com -v
```

### Cron example

Check every 15 minutes:

```cron
*/15 * * * * /path/to/venv/bin/kaktus-promo user@example.com
```

## Development

```bash
make setup    # Install dependencies
make lint     # Run ruff and mypy
make format   # Format code with ruff
make test     # Run tests
make clean    # Remove venv and caches
```

## License

MIT
