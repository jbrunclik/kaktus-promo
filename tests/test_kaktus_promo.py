#!/usr/bin/env python3

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import quote

import requests

import kaktus_promo
from kaktus_promo import HttpClient, State, TimeRange

TEST_RECIPIENTS = ["test@example.com"]
TEST_STATE_FILE = "/tmp/test_state.json"


class TestTimeRange(unittest.TestCase):

    def test_format_same_day(self):
        """Test formatting when start and end are on the same day."""
        tr = TimeRange(
            start=datetime(2025, 6, 4, 10, 0),
            end=datetime(2025, 6, 4, 18, 30),
        )
        self.assertEqual(tr.format(), "04/06/2025 10:00-18:30")

    def test_format_different_days(self):
        """Test formatting when start and end are on different days."""
        tr = TimeRange(
            start=datetime(2025, 6, 3, 0, 0),
            end=datetime(2025, 6, 4, 23, 59),
        )
        self.assertEqual(tr.format(), "03/06/2025 00:00 - 04/06/2025 23:59")


class TestState(unittest.TestCase):

    def setUp(self):
        self.mock_logger = Mock()

    def test_load_nonexistent_file(self):
        """Test loading state when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            state = State.load(path, self.mock_logger)
            self.assertIsNone(state.tc_url)
            self.assertIsNone(state.promo_date)

    def test_load_valid_file(self):
        """Test loading state from a valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "tc_url": "https://example.com/tc.pdf",
                "promo_date": "2025-06-04T00:00:00",
                "last_checked": "2025-06-01T12:00:00",
                "last_updated": "2025-06-01T12:00:00",
            }, f)
            path = Path(f.name)

        try:
            state = State.load(path, self.mock_logger)
            self.assertEqual(state.tc_url, "https://example.com/tc.pdf")
            self.assertEqual(state.promo_date, datetime(2025, 6, 4))
        finally:
            path.unlink()

    def test_load_invalid_json(self):
        """Test loading state with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json")
            path = Path(f.name)

        try:
            state = State.load(path, self.mock_logger)
            self.assertIsNone(state.tc_url)
        finally:
            path.unlink()

    def test_save_and_load_roundtrip(self):
        """Test saving and loading state."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            state = State(
                tc_url="https://example.com/tc.pdf",
                promo_date=datetime(2025, 6, 4),
            )
            state.save(path, self.mock_logger)

            loaded = State.load(path, self.mock_logger)
            self.assertEqual(loaded.tc_url, state.tc_url)
            self.assertEqual(loaded.promo_date, state.promo_date)
        finally:
            path.unlink()

    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = State(
            tc_url="https://example.com/tc.pdf",
            promo_date=datetime(2025, 6, 4),
            last_checked=datetime(2025, 6, 1, 12, 0),
            last_updated=datetime(2025, 6, 1, 12, 0),
        )
        d = state.to_dict()
        self.assertEqual(d["tc_url"], "https://example.com/tc.pdf")
        self.assertEqual(d["promo_date"], "2025-06-04T00:00:00")


class TestHttpClient(unittest.TestCase):

    def test_get_success(self):
        """Test successful GET request."""
        mock_response = Mock()
        mock_response.content = b"test content"

        client = HttpClient()
        with patch.object(client.session, "get", return_value=mock_response):
            response = client.get("https://example.com")

        self.assertEqual(response.content, b"test content")
        mock_response.raise_for_status.assert_called_once()


class TestKaktusPromo(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.tc_filename = "OP-Odmena-za-dobiti-FB_04062025.pdf"
        self.doc_path_prefix = "/api/documents/file/"

        def build_download_url(filename):
            encoded_path = quote(
                f"{self.doc_path_prefix}{filename}",
                safe=""
            )
            return (
                "https://www.mujkaktus.cz/api/download?"
                f"docUrl={encoded_path}&filename={filename}"
            )

        self.make_download_url = build_download_url
        self.full_tc_url = self.make_download_url(self.tc_filename)

        long_href = (
            "/api/download?docUrl=%2Fapi%2Fdocuments%2Ffile%2F"
            f"{self.tc_filename}&filename={self.tc_filename}"
        )
        self.sample_html_with_tc = f"""
        <html>
            <body>
                <a href="{long_href}">
                    Celé podmínky v PDF
                </a>
            </body>
        </html>
        """.encode()

        self.sample_html_without_tc = b"""
        <html>
            <body>
                <p>No terms and conditions here</p>
            </body>
        </html>
        """

        self.sample_state = {
            "tc_url": self.make_download_url(
                "OP-Odmena-za-dobiti-FB_01012025.pdf"
            ),
            "promo_date": "2025-01-01T00:00:00",
            "last_checked": "2025-01-15T10:00:00",
            "last_updated": "2025-01-01T09:00:00"
        }

        self.mock_logger = Mock()
        self.mock_client = Mock(spec=HttpClient)

    @patch.object(HttpClient, "get")
    def test_get_promo_page_success(self, mock_get):
        """Test successful page fetching."""
        mock_response = Mock()
        mock_response.content = self.sample_html_with_tc
        mock_get.return_value = mock_response

        client = HttpClient()
        result = kaktus_promo._get_promo_page(client, self.mock_logger)

        self.assertEqual(result, self.sample_html_with_tc)

    @patch.object(HttpClient, "get")
    def test_get_promo_page_failure(self, mock_get):
        """Test page fetching failure."""
        mock_get.side_effect = requests.RequestException("Network error")

        client = HttpClient()
        with self.assertRaises(requests.RequestException):
            kaktus_promo._get_promo_page(client, self.mock_logger)

    def test_extract_tc_url_found(self):
        """Test T&C URL extraction when URL is present."""
        result = kaktus_promo._extract_tc_url(
            self.sample_html_with_tc, self.mock_logger
        )

        self.assertEqual(result, self.full_tc_url)

    def test_extract_tc_url_not_found(self):
        """Test T&C URL extraction when URL is not present."""
        result = kaktus_promo._extract_tc_url(
            self.sample_html_without_tc, self.mock_logger
        )

        self.assertIsNone(result)

    def test_extract_tc_url_malformed_content(self):
        """Test T&C URL extraction with malformed content."""
        malformed_content = b"\xff\xfe\x00\x00invalid"

        result = kaktus_promo._extract_tc_url(malformed_content, self.mock_logger)

        self.assertIsNone(result)

    def test_extract_promo_date_valid(self):
        """Test promo date extraction with valid URL."""
        result = kaktus_promo._extract_promo_date(
            self.full_tc_url, self.mock_logger
        )

        expected_date = datetime(2025, 6, 4)
        self.assertEqual(result, expected_date)

    def test_extract_promo_date_invalid_date(self):
        """Test promo date extraction with invalid date."""
        tc_url = self.make_download_url("OP-Odmena-za-dobiti-FB_99999999.pdf")

        result = kaktus_promo._extract_promo_date(tc_url, self.mock_logger)

        self.assertIsNone(result)

    def test_extract_promo_date_no_date_pattern(self):
        """Test promo date extraction when no date pattern is found."""
        tc_url = self.make_download_url("other.pdf")

        result = kaktus_promo._extract_promo_date(tc_url, self.mock_logger)

        self.assertIsNone(result)

    def test_extract_promo_date_none_input(self):
        """Test promo date extraction with None input."""
        result = kaktus_promo._extract_promo_date(None, self.mock_logger)

        self.assertIsNone(result)

    @patch.object(HttpClient, "get")
    def test_fetch_tc_pdf_not_found(self, mock_get):
        """Return None and warn when promo PDF is missing."""
        http_error = requests.HTTPError("Not Found")
        http_error.response = Mock(status_code=404)
        mock_get.side_effect = http_error

        client = HttpClient()
        result = kaktus_promo._fetch_tc_pdf(client, self.full_tc_url, self.mock_logger)

        self.assertIsNone(result)
        self.mock_logger.warning.assert_called_once_with(
            "Promo PDF not available at %s (404)",
            self.full_tc_url
        )

    @patch("kaktus_promo.Popen")
    def test_send_email_with_time_range(self, mock_popen):
        """Test sending email with time range."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        tc_url = self.full_tc_url
        promo_date = datetime(2025, 6, 4)
        time_range = TimeRange(
            datetime(2025, 6, 4, 10, 0),
            datetime(2025, 6, 4, 18, 30),
        )

        kaktus_promo._send_email(
            TEST_RECIPIENTS, tc_url, promo_date, self.mock_logger,
            time_range
        )

        mock_popen.assert_called_once_with(
            ["/usr/sbin/sendmail", "-t", "-oi"], stdin=kaktus_promo.PIPE
        )
        mock_process.communicate.assert_called_once()

    @patch("kaktus_promo.Popen")
    def test_send_email_first_time(self, mock_popen):
        """Test sending email for first time detection."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        tc_url = self.full_tc_url
        promo_date = datetime(2025, 6, 4)

        kaktus_promo._send_email(
            TEST_RECIPIENTS, tc_url, promo_date, self.mock_logger
        )

        mock_popen.assert_called_once()
        mock_process.communicate.assert_called_once()

    @patch("kaktus_promo.Popen")
    def test_send_email_sendmail_failure(self, mock_popen):
        """Test email sending when sendmail fails."""
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_popen.return_value = mock_process

        tc_url = self.full_tc_url
        promo_date = datetime(2025, 6, 4)

        kaktus_promo._send_email(
            TEST_RECIPIENTS, tc_url, promo_date, self.mock_logger
        )

        mock_popen.assert_called_once()

    @patch("kaktus_promo._send_email")
    @patch("kaktus_promo._extract_promo_time_range")
    @patch("kaktus_promo._fetch_tc_pdf")
    @patch.object(State, "load")
    @patch.object(State, "save")
    @patch("kaktus_promo._extract_promo_date")
    @patch("kaktus_promo._extract_tc_url")
    @patch("kaktus_promo._get_promo_page")
    def test_main_new_promo_detected(self, mock_get_page, mock_extract_url,
                                     mock_extract_date, mock_save_state,
                                     mock_load_state, mock_fetch_pdf,
                                     mock_extract_range, mock_send_email):
        """Test main function when new promo is detected."""
        mock_get_page.return_value = self.sample_html_with_tc
        mock_extract_url.return_value = self.full_tc_url
        mock_extract_date.return_value = datetime(2025, 6, 4)
        mock_load_state.return_value = State(
            tc_url=(
                "https://www.mujkaktus.cz/api/documents/file/"
                "OP-Odmena-za-dobiti-FB_01012025.pdf"
            ),
            promo_date=datetime(2025, 1, 1),
        )
        mock_fetch_pdf.return_value = b"%PDF-mock"
        mock_extract_range.return_value = TimeRange(
            datetime(2025, 6, 4, 0, 0),
            datetime(2025, 6, 4, 23, 59)
        )

        kaktus_promo.main(TEST_RECIPIENTS, TEST_STATE_FILE, verbose=True)

        mock_fetch_pdf.assert_called_once()
        mock_extract_range.assert_called_once()
        mock_send_email.assert_called_once()
        mock_save_state.assert_called_once()

    @patch("kaktus_promo._send_email")
    @patch("kaktus_promo._extract_promo_time_range")
    @patch("kaktus_promo._fetch_tc_pdf")
    @patch.object(State, "load")
    @patch("kaktus_promo._extract_promo_date")
    @patch("kaktus_promo._extract_tc_url")
    @patch("kaktus_promo._get_promo_page")
    @patch("kaktus_promo.Path")
    def test_main_no_change_detected(self, mock_path, mock_get_page, mock_extract_url,
                                     mock_extract_date, mock_load_state,
                                     mock_fetch_pdf, mock_extract_range,
                                     mock_send_email):
        """Test main function when no change is detected."""
        mock_get_page.return_value = self.sample_html_with_tc
        mock_extract_url.return_value = self.full_tc_url
        mock_extract_date.return_value = datetime(2025, 6, 4)
        # Use the full URL so the filename matches
        mock_load_state.return_value = State(
            tc_url=self.full_tc_url,
            promo_date=datetime(2025, 6, 4),
        )
        mock_path.return_value.write_text = Mock()

        kaktus_promo.main(TEST_RECIPIENTS, TEST_STATE_FILE, verbose=True)

        mock_fetch_pdf.assert_not_called()
        mock_extract_range.assert_not_called()
        mock_send_email.assert_not_called()

    @patch("kaktus_promo._send_email")
    @patch("kaktus_promo._extract_promo_time_range")
    @patch("kaktus_promo._fetch_tc_pdf")
    @patch.object(State, "load")
    @patch("kaktus_promo._extract_promo_date")
    @patch("kaktus_promo._extract_tc_url")
    @patch("kaktus_promo._get_promo_page")
    @patch("kaktus_promo.Path")
    def test_main_pdf_missing(self, mock_path, mock_get_page, mock_extract_url,
                              mock_extract_date, mock_load_state,
                              mock_fetch_pdf, mock_extract_range,
                              mock_send_email):
        """Skip notifications when the promo PDF is unavailable."""
        mock_get_page.return_value = self.sample_html_with_tc
        mock_extract_url.return_value = self.full_tc_url
        mock_extract_date.return_value = datetime(2025, 6, 4)
        mock_load_state.return_value = State(
            tc_url=(
                "https://www.mujkaktus.cz/api/documents/file/"
                "OP-Odmena-za-dobiti-FB_01012025.pdf"
            ),
            promo_date=datetime(2025, 1, 1),
        )
        mock_fetch_pdf.return_value = None
        mock_path.return_value.write_text = Mock()

        kaktus_promo.main(TEST_RECIPIENTS, TEST_STATE_FILE, verbose=True)

        mock_fetch_pdf.assert_called_once()
        mock_extract_range.assert_not_called()
        mock_send_email.assert_not_called()

    @patch("kaktus_promo._extract_tc_url")
    @patch("kaktus_promo._get_promo_page")
    def test_main_no_tc_url_found(self, mock_get_page, mock_extract_url):
        """Test main function when no T&C URL is found."""
        mock_get_page.return_value = self.sample_html_without_tc
        mock_extract_url.return_value = None

        kaktus_promo.main(TEST_RECIPIENTS, TEST_STATE_FILE, verbose=True)

        mock_extract_url.assert_called_once()

    @patch("kaktus_promo._get_promo_page")
    def test_main_network_error(self, mock_get_page):
        """Test main function when network error occurs."""
        mock_get_page.side_effect = Exception("Network error")

        with self.assertRaises(Exception):
            kaktus_promo.main(TEST_RECIPIENTS, TEST_STATE_FILE, verbose=True)

    def test_date_extraction_edge_cases(self):
        """Test date extraction with various edge cases."""
        test_cases = [
            (self.make_download_url("OP-Odmena-za-dobiti-FB_29022024.pdf"),
             datetime(2024, 2, 29)),
            (self.make_download_url("OP-Odmena-za-dobiti-FB_31122025.pdf"),
             datetime(2025, 12, 31)),
            (self.make_download_url("OP-Odmena-za-dobiti-FB_01012000.pdf"),
             datetime(2000, 1, 1)),
        ]

        for tc_url, expected_date in test_cases:
            with self.subTest(tc_url=tc_url):
                result = kaktus_promo._extract_promo_date(
                    tc_url, self.mock_logger
                )
                self.assertEqual(result, expected_date)

    def test_invalid_date_formats(self):
        """Test date extraction with invalid date formats."""
        invalid_cases = [
            self.make_download_url("OP-Odmena-za-dobiti-FB_32012025.pdf"),
            self.make_download_url("OP-Odmena-za-dobiti-FB_01132025.pdf"),
            self.make_download_url("OP-Odmena-za-dobiti-FB_29022023.pdf"),
            self.make_download_url("OP-Odmena-za-dobiti-FB_1234567.pdf"),
            self.make_download_url("OP-Odmena-za-dobiti-FB_abcd2025.pdf"),
        ]

        for tc_url in invalid_cases:
            with self.subTest(tc_url=tc_url):
                result = kaktus_promo._extract_promo_date(
                    tc_url, self.mock_logger
                )
                self.assertIsNone(result)

    def test_parse_time_range_from_text(self):
        """Test parsing time range from standard text layout."""
        text = "Akce probiha od 3. 6. 2025 0:00 do 4. 6. 2025 23:59."

        result = kaktus_promo._parse_time_range_from_text(text, self.mock_logger)

        expected = TimeRange(
            datetime(2025, 6, 3, 0, 0),
            datetime(2025, 6, 4, 23, 59)
        )

        self.assertEqual(result, expected)

    def test_parse_time_range_with_time_before_date(self):
        """Test parsing when time precedes the date."""
        text = (
            "Akce probiha od 0:00 hod dne 3. 6. 2025 do 23:59:59 hod dne"
            " 4. 6. 2025."
        )

        result = kaktus_promo._parse_time_range_from_text(text, self.mock_logger)

        expected = TimeRange(
            datetime(2025, 6, 3, 0, 0),
            datetime(2025, 6, 4, 23, 59, 59)
        )

        self.assertEqual(result, expected)

    def test_setup_logging_verbose(self):
        """Test logging setup in verbose mode."""
        logger = kaktus_promo._setup_logging(verbose=True)
        self.assertEqual(logger.level, 0)

    def test_setup_logging_quiet(self):
        """Test logging setup in quiet mode."""
        logger = kaktus_promo._setup_logging(verbose=False)
        self.assertEqual(logger.level, 0)


class TestKaktusPromoIntegration(unittest.TestCase):
    """Integration tests using real file operations."""

    def setUp(self):
        """Set up temporary file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".json"
        )
        self.state_file = self.temp_file.name
        self.mock_logger = Mock()

    def tearDown(self):
        """Clean up temporary file."""
        Path(self.temp_file.name).unlink(missing_ok=True)

    def test_state_persistence_integration(self):
        """Test that state is properly saved and loaded."""
        filename = "OP-Odmena-za-dobiti-FB_04062025.pdf"
        encoded_path = quote(f"/api/documents/file/{filename}", safe="")
        tc_url = (
            "https://www.mujkaktus.cz/api/download?"
            f"docUrl={encoded_path}&filename={filename}"
        )
        promo_date = datetime(2025, 6, 4)

        state = State(tc_url=tc_url, promo_date=promo_date)
        state.save(Path(self.state_file), self.mock_logger)

        loaded_state = State.load(Path(self.state_file), self.mock_logger)

        self.assertEqual(loaded_state.tc_url, tc_url)
        self.assertEqual(loaded_state.promo_date, promo_date)
        self.assertIsNotNone(loaded_state.last_checked)
        self.assertIsNotNone(loaded_state.last_updated)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    unittest.main(verbosity=2)
