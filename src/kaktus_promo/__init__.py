#!/usr/bin/env python3
"""Monitor mujkaktus.cz for credit-doubling promotional offers."""

from __future__ import annotations

import io
import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from subprocess import PIPE, Popen
from typing import NamedTuple, Self
from urllib.parse import parse_qs, quote, urljoin, urlparse

import requests
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROMO_URL = "https://www.mujkaktus.cz/chces-pridat"
DEFAULT_STATE_FILE = "/var/tmp/mujkaktus_state.json"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36"
)
DATE_PATTERN = re.compile(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})")
TIME_PATTERN = re.compile(r"(\d{1,2})[:.](\d{2})(?::(\d{2}))?")


class TimeRange(NamedTuple):
    """Promo time window with start and end datetimes."""

    start: datetime
    end: datetime

    def format(self) -> str:
        """Return a compact string representing the time range."""
        if self.start.date() == self.end.date():
            return f"{self.start:%d/%m/%Y %H:%M}-{self.end:%H:%M}"
        return f"{self.start:%d/%m/%Y %H:%M} - {self.end:%d/%m/%Y %H:%M}"


@dataclass
class State:
    """Persistent state tracking the last seen promo."""

    tc_url: str | None = None
    promo_date: datetime | None = None
    last_checked: datetime | None = None
    last_updated: datetime | None = None

    @classmethod
    def load(cls, path: Path, logger: logging.Logger) -> Self:
        """Load state from a JSON file."""
        try:
            if path.exists():
                data = json.loads(path.read_text())
                logger.info("Loaded previous state: %s", data)
                return cls(
                    tc_url=data.get("tc_url"),
                    promo_date=(
                        datetime.fromisoformat(data["promo_date"])
                        if data.get("promo_date")
                        else None
                    ),
                    last_checked=(
                        datetime.fromisoformat(data["last_checked"])
                        if data.get("last_checked")
                        else None
                    ),
                    last_updated=(
                        datetime.fromisoformat(data["last_updated"])
                        if data.get("last_updated")
                        else None
                    ),
                )
            logger.info("No previous state file found")
            return cls()
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.error("Error loading state file: %s", e)
            return cls()

    def save(self, path: Path, logger: logging.Logger) -> None:
        """Save state to a JSON file."""
        try:
            now = datetime.now()
            data = {
                "tc_url": self.tc_url,
                "promo_date": self.promo_date.isoformat() if self.promo_date else None,
                "last_checked": now.isoformat(),
                "last_updated": now.isoformat(),
            }
            path.write_text(json.dumps(data, indent=2))
            logger.info("Saved state: %s", data)
        except OSError as e:
            logger.error("Error saving state file: %s", e)
            raise

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tc_url": self.tc_url,
            "promo_date": self.promo_date.isoformat() if self.promo_date else None,
            "last_checked": (
                self.last_checked.isoformat() if self.last_checked else None
            ),
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
        }


@dataclass
class HttpClient:
    """Configured HTTP client with retry logic."""

    session: requests.Session = field(default_factory=requests.Session)
    timeout: int = 30

    def __post_init__(self) -> None:
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.headers["User-Agent"] = USER_AGENT

    def get(self, url: str) -> requests.Response:
        """Perform a GET request with configured retries."""
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response


def _get_tc_filename(tc_url: str | None) -> str | None:
    """Return the promo filename embedded in the URL."""
    if not tc_url:
        return None

    parsed = urlparse(tc_url)
    if filename := parse_qs(parsed.query).get("filename"):
        return filename[0]

    return Path(parsed.path).name


def _setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging based on verbosity level."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def _get_promo_page(client: HttpClient, logger: logging.Logger) -> bytes:
    """Fetch the promo page content with retry mechanism."""
    try:
        response = client.get(PROMO_URL)
        logger.info(
            "Successfully fetched page, content length: %s",
            len(response.content),
        )
        return response.content
    except requests.RequestException as e:
        logger.error("Failed to fetch promo page: %s", e)
        raise


def _extract_tc_url(promo_page: bytes, logger: logging.Logger) -> str | None:
    """Extract the Terms & Conditions URL from the page content."""
    try:
        content_str = promo_page.decode("utf-8", errors="ignore")

        if href_match := re.search(
            r'href="([^"]*/api/download\?docUrl=[^"]+\.pdf[^"]*)"',
            content_str,
        ):
            href: str = href_match.group(1).replace("&amp;", "&")
            full_url = urljoin(PROMO_URL, href)
            logger.info("Found T&C URL: %s", full_url)
            return full_url

        if doc_url_match := re.search(r"docUrl=([^\"&]+)", content_str):
            encoded_path = doc_url_match.group(1)
            filename_match = re.search(r"filename=([^\"&]+)", content_str)
            filename = filename_match.group(1) if filename_match else ""
            fallback_href = f"/api/download?docUrl={encoded_path}"
            if filename:
                fallback_href += f"&filename={filename}"
            full_url = urljoin(PROMO_URL, fallback_href)
            logger.info("Found T&C URL: %s", full_url)
            return full_url

        tc_pattern = r'OP-Odmena-za-dobiti-[^"&\\]+\.pdf'
        if matches := re.findall(tc_pattern, content_str):
            filename = matches[0]
            encoded_path = quote(f"/api/documents/file/{filename}", safe="")
            full_url = urljoin(
                PROMO_URL,
                f"/api/download?docUrl={encoded_path}&filename={filename}",
            )
            logger.info("Found T&C URL: %s", full_url)
            return full_url

        logger.warning("No T&C URL found in page content")
        return None

    except Exception as e:
        logger.error("Error extracting T&C URL: %s", e)
        return None


def _extract_promo_date(
    tc_url: str | None, logger: logging.Logger
) -> datetime | None:
    """Extract the date from the T&C URL."""
    if not tc_url:
        return None

    try:
        filename = _get_tc_filename(tc_url)
        if not filename:
            return None

        if not (match := re.search(r"FB_(\d{8})", filename)):
            return None

        promo_date = datetime.strptime(match.group(1), "%d%m%Y")
        logger.info("Extracted promo date: %s", promo_date.strftime("%Y-%m-%d"))
        return promo_date
    except ValueError as exc:
        logger.error("Failed to parse promo date from '%s': %s", tc_url, exc)
        return None


def _fetch_tc_pdf(
    client: HttpClient, tc_url: str | None, logger: logging.Logger
) -> bytes | None:
    """Download the promo Terms & Conditions PDF."""
    if not tc_url:
        return None

    logger.info("Fetching promo PDF: %s", tc_url)

    try:
        response = client.get(tc_url)
        logger.info(
            "Fetched promo PDF successfully, size: %s bytes",
            len(response.content),
        )
        return response.content
    except requests.HTTPError as exc:
        if getattr(exc.response, "status_code", None) == 404:
            logger.warning("Promo PDF not available at %s (404)", tc_url)
            return None
        logger.error("Failed to download promo PDF: %s", exc)
        return None
    except requests.RequestException as exc:
        logger.error("Failed to download promo PDF: %s", exc)
        return None


def _parse_time_range_from_text(
    text: str | None, logger: logging.Logger
) -> TimeRange | None:
    """Extract promo time range from plain text."""
    if not text:
        return None

    normalized = re.sub(r"\s+", " ", text)
    date_matches = DATE_PATTERN.findall(normalized)
    time_matches = TIME_PATTERN.findall(normalized)

    if len(date_matches) < 2 or len(time_matches) < 2:
        logger.warning("Unable to detect two promo datetimes in promo PDF text")
        return None

    def _build_dt(
        date_parts: tuple[str, str, str], time_parts: tuple[str, str, str]
    ) -> datetime:
        day, month, year = map(int, date_parts)
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2] or 0)
        return datetime(year, month, day, hour, minute, second)

    try:
        start_dt = _build_dt(date_matches[0], time_matches[0])
        end_dt = _build_dt(date_matches[1], time_matches[1])
    except ValueError as exc:
        logger.warning("Invalid datetime extracted from promo PDF: %s", exc)
        return None

    return TimeRange(start_dt, end_dt)


def _extract_promo_time_range(
    pdf_bytes: bytes | None, logger: logging.Logger
) -> TimeRange | None:
    """Extract the promo time range from a PDF file."""
    if not pdf_bytes:
        return None

    try:
        pdf_file = io.BytesIO(pdf_bytes)
        # Suppress pypdf warnings (logged to root logger and via warnings module)
        pypdf_logger = logging.getLogger("pypdf")
        original_level = pypdf_logger.level
        pypdf_logger.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = PdfReader(pdf_file)
        finally:
            pypdf_logger.setLevel(original_level)

        text_chunks = [
            page_text
            for page in reader.pages
            if (page_text := page.extract_text())
        ]
        combined_text = "\n".join(text_chunks)

        if not combined_text.strip():
            logger.warning("No text extracted from promo PDF")
            return None

        time_range = _parse_time_range_from_text(combined_text, logger)

        if time_range:
            logger.info(
                "Parsed promo time range: %s - %s",
                time_range.start.isoformat(),
                time_range.end.isoformat(),
            )
        else:
            logger.warning("Promo time range not found in PDF")

        return time_range
    except Exception as exc:
        logger.error("Error extracting promo time range from PDF: %s", exc)
        return None


def _send_email(
    recipients: list[str],
    tc_url: str,
    promo_date: datetime | None,
    logger: logging.Logger,
    time_range: TimeRange | None = None,
) -> None:
    """Send email notification about promo change."""
    try:
        if time_range:
            # Format: "4. 6. 2025, 10:00-18:30" or "3. 6. 10:00 - 4. 6. 18:30"
            if time_range.start.date() == time_range.end.date():
                when_str = (
                    f"{time_range.start:%-d. %-m. %Y}, "
                    f"{time_range.start:%H:%M}-{time_range.end:%H:%M}"
                )
                subject_when = (
                    f"{time_range.start:%-d.%-m.} "
                    f"{time_range.start:%H:%M}-{time_range.end:%H:%M}"
                )
            else:
                when_str = (
                    f"{time_range.start:%-d. %-m. %Y %H:%M} - "
                    f"{time_range.end:%-d. %-m. %Y %H:%M}"
                )
                subject_when = (
                    f"{time_range.start:%-d.%-m.} {time_range.start:%H:%M} - "
                    f"{time_range.end:%-d.%-m.} {time_range.end:%H:%M}"
                )
        elif promo_date:
            when_str = f"{promo_date:%-d. %-m. %Y}"
            subject_when = f"{promo_date:%-d.%-m.%Y}"
        else:
            when_str = "neznámý termín"
            subject_when = "neznámý termín"

        msg = EmailMessage()
        msg["From"] = recipients[0]
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"Dobíječka {subject_when}"

        body = "\n".join([
            "Nová Dobíječka na mujkaktus.cz!",
            "",
            f"Kdy: {when_str}",
            f"Web: {PROMO_URL}",
            "",
            "---",
            f"Podmínky: {tc_url}",
        ])

        msg.set_content(body)

        with Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=PIPE) as proc:
            proc.communicate(msg.as_bytes())
            if proc.returncode == 0:
                logger.info("Email sent successfully to %s", recipients)
            else:
                logger.error("Failed to send email, return code: %s", proc.returncode)

    except Exception as e:
        logger.error("Error sending email: %s", e)
        raise


def main(
    recipients: list[str],
    state_file: str = DEFAULT_STATE_FILE,
    verbose: bool = False,
) -> None:
    """Main function to check for promo changes."""
    logger = _setup_logging(verbose)
    client = HttpClient()
    state_path = Path(state_file)

    try:
        logger.info("Starting promo check...")

        promo_page = _get_promo_page(client, logger)
        tc_url = _extract_tc_url(promo_page, logger)
        promo_date = _extract_promo_date(tc_url, logger)

        if not tc_url:
            logger.error("Could not extract T&C URL, skipping check")
            return

        state = State.load(state_path, logger)
        last_identifier = _get_tc_filename(state.tc_url)
        current_identifier = _get_tc_filename(tc_url)

        if current_identifier != last_identifier:
            logger.info("T&C URL changed from '%s' to '%s'", state.tc_url, tc_url)

            pdf_bytes = _fetch_tc_pdf(client, tc_url, logger)
            if not pdf_bytes:
                logger.warning(
                    "Promo PDF unavailable; treating detected change as provisional"
                )
                state.tc_url = tc_url
                state.last_checked = datetime.now()
                state_path.write_text(json.dumps(state.to_dict(), indent=2))
                return

            time_range = _extract_promo_time_range(pdf_bytes, logger)

            _send_email(
                recipients,
                tc_url,
                promo_date,
                logger,
                time_range,
            )

            new_state = State(tc_url=tc_url, promo_date=promo_date)
            new_state.save(state_path, logger)

        else:
            logger.info("No change in T&C URL detected")

            if state.tc_url != tc_url:
                state.tc_url = tc_url
            state.last_checked = datetime.now()
            state_path.write_text(json.dumps(state.to_dict(), indent=2))

        logger.info("Promo check completed successfully")

    except Exception as e:
        logger.error("Error in main function: %s", e)
        raise
