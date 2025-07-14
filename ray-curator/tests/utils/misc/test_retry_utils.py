"""Test suite for retry_utils module."""

import time
from unittest.mock import Mock, patch

import pytest

from ray_curator.utils.misc.retry_utils import do_with_retries


class TestDoWithRetries:
    """Test suite for do_with_retries function."""

    def test_successful_execution_first_try(self) -> None:
        """Test successful execution on first attempt."""
        mock_func = Mock(return_value="success")

        result = do_with_retries(mock_func)

        assert result == "success"
        mock_func.assert_called_once()

    def test_successful_execution_with_return_value(self) -> None:
        """Test successful execution with different return types."""
        test_cases = [
            42,
            "string_result",
            [1, 2, 3],
            {"key": "value"},
            None,
        ]

        for expected_result in test_cases:
            mock_func = Mock(return_value=expected_result)
            result = do_with_retries(mock_func)
            assert result == expected_result

    def test_retry_on_exception_then_success(self) -> None:
        """Test retry after exception, then success."""
        mock_func = Mock(side_effect=[ValueError("first fail"), "success"])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep") as mock_sleep,
            patch("ray_curator.utils.misc.retry_utils.logger") as mock_logger,
        ):
            result = do_with_retries(mock_func)

            assert result == "success"
            assert mock_func.call_count == 2
            mock_sleep.assert_called_once_with(2.0)  # backoff_factor^1
            mock_logger.warning.assert_called_once()
            assert "Attempt 1/5 failed" in mock_logger.warning.call_args[0][0]

    def test_retry_multiple_times_then_success(self) -> None:
        """Test multiple retries before success."""
        mock_func = Mock(side_effect=[
            ValueError("fail 1"),
            RuntimeError("fail 2"),
            ConnectionError("fail 3"),
            "success"
        ])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep") as mock_sleep,
            patch("ray_curator.utils.misc.retry_utils.logger") as mock_logger,
        ):
            result = do_with_retries(mock_func)

            assert result == "success"
            assert mock_func.call_count == 4

            # Check sleep calls for exponential backoff
            expected_sleeps = [2.0, 4.0, 8.0]  # 2^1, 2^2, 2^3
            actual_sleeps = [call[0][0] for call in mock_sleep.call_args_list]
            assert actual_sleeps == expected_sleeps

            # Check that logger was called 3 times
            assert mock_logger.warning.call_count == 3

    def test_max_attempts_reached(self) -> None:
        """Test that exception is raised when max attempts reached."""
        mock_func = Mock(side_effect=ValueError("always fails"))

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep"),
            patch("ray_curator.utils.misc.retry_utils.logger"),
        ):
            with pytest.raises(ValueError, match="always fails"):
                do_with_retries(mock_func, max_attempts=3)

            assert mock_func.call_count == 3

    def test_custom_max_attempts(self) -> None:
        """Test custom max_attempts parameter."""
        mock_func = Mock(side_effect=ValueError("always fails"))

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep"),
            patch("ray_curator.utils.misc.retry_utils.logger"),
        ):
            with pytest.raises(ValueError, match="always fails"):
                do_with_retries(mock_func, max_attempts=2)

            assert mock_func.call_count == 2

    def test_custom_backoff_factor(self) -> None:
        """Test custom backoff_factor parameter."""
        mock_func = Mock(side_effect=[ValueError("fail 1"), ValueError("fail 2"), "success"])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep") as mock_sleep,
            patch("ray_curator.utils.misc.retry_utils.logger"),
        ):
            result = do_with_retries(mock_func, backoff_factor=3.0)

            assert result == "success"
            expected_sleeps = [3.0, 9.0]  # 3^1, 3^2
            actual_sleeps = [call[0][0] for call in mock_sleep.call_args_list]
            assert actual_sleeps == expected_sleeps

    def test_max_wait_time_limit(self) -> None:
        """Test that wait time is capped by max_wait_time_s."""
        mock_func = Mock(side_effect=[ValueError("fail 1"), ValueError("fail 2"), "success"])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep") as mock_sleep,
            patch("ray_curator.utils.misc.retry_utils.logger"),
        ):
            result = do_with_retries(
                mock_func,
                backoff_factor=10.0,
                max_wait_time_s=5.0
            )

            assert result == "success"
            # Both sleeps should be capped at 5.0
            expected_sleeps = [5.0, 5.0]  # min(10^1, 5.0), min(10^2, 5.0)
            actual_sleeps = [call[0][0] for call in mock_sleep.call_args_list]
            assert actual_sleeps == expected_sleeps

    def test_specific_exceptions_to_retry(self) -> None:
        """Test retrying only specific exception types."""
        mock_func = Mock(side_effect=[ValueError("retryable"), "success"])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep"),
            patch("ray_curator.utils.misc.retry_utils.logger"),
        ):
            result = do_with_retries(
                mock_func,
                exceptions_to_retry=[ValueError, RuntimeError]
            )

            assert result == "success"
            assert mock_func.call_count == 2

    def test_exception_not_in_retry_list(self) -> None:
        """Test that exceptions not in retry list are not retried."""
        mock_func = Mock(side_effect=ConnectionError("not retryable"))

        with pytest.raises(ConnectionError, match="not retryable"):
            do_with_retries(
                mock_func,
                exceptions_to_retry=[ValueError, RuntimeError]
            )

        mock_func.assert_called_once()

    def test_named_retry_logging(self) -> None:
        """Test that named retries include name in log messages."""
        mock_func = Mock(side_effect=[ValueError("fail"), "success"])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep"),
            patch("ray_curator.utils.misc.retry_utils.logger") as mock_logger,
        ):
            result = do_with_retries(mock_func, name="test_operation")

            assert result == "success"
            mock_logger.warning.assert_called_once()
            log_message = mock_logger.warning.call_args[0][0]
            assert "test_operation - Attempt 1/5 failed" in log_message

    def test_unnamed_retry_logging(self) -> None:
        """Test that unnamed retries don't include name in log messages."""
        mock_func = Mock(side_effect=[ValueError("fail"), "success"])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep"),
            patch("ray_curator.utils.misc.retry_utils.logger") as mock_logger,
        ):
            result = do_with_retries(mock_func)

            assert result == "success"
            mock_logger.warning.assert_called_once()
            log_message = mock_logger.warning.call_args[0][0]
            assert "Attempt 1/5 failed" in log_message
            assert " - " not in log_message.split("Attempt")[0]

    def test_log_message_format(self) -> None:
        """Test the format of log messages."""
        mock_func = Mock(side_effect=[ValueError("test error"), "success"])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep"),
            patch("ray_curator.utils.misc.retry_utils.logger") as mock_logger,
        ):
            do_with_retries(mock_func, name="test_op")

            log_message = mock_logger.warning.call_args[0][0]
            assert "test_op - Attempt 1/5 failed with error: test error" in log_message
            assert "Retrying in 2 seconds" in log_message

    def test_default_exceptions_to_retry(self) -> None:
        """Test that default behavior retries all exceptions."""
        exception_types = [ValueError, RuntimeError, ConnectionError, KeyError]

        for exception_type in exception_types:
            mock_func = Mock(side_effect=[exception_type("fail"), "success"])

            with (
                patch("ray_curator.utils.misc.retry_utils.time.sleep"),
                patch("ray_curator.utils.misc.retry_utils.logger"),
            ):
                result = do_with_retries(mock_func)

                assert result == "success"
                assert mock_func.call_count == 2

    def test_empty_exceptions_to_retry_list(self) -> None:
        """Test behavior with empty exceptions_to_retry list."""
        mock_func = Mock(side_effect=ValueError("should not retry"))

        with pytest.raises(ValueError, match="should not retry"):
            do_with_retries(mock_func, exceptions_to_retry=[])

        mock_func.assert_called_once()

    def test_function_with_arguments_via_lambda(self) -> None:
        """Test retrying a function with arguments using lambda."""
        def func_with_args(x: int, y: str) -> str:
            if x < 5:
                msg = f"x too small: {x}"
                raise ValueError(msg)
            return f"result: {x}, {y}"

        call_count = 0
        def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "first attempt fails"
                raise ValueError(msg)
            return func_with_args(10, "test")

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep"),
            patch("ray_curator.utils.misc.retry_utils.logger"),
        ):
            result = do_with_retries(failing_func)

            assert result == "result: 10, test"
            assert call_count == 2

    def test_sleep_duration_calculation(self) -> None:
        """Test that sleep duration is calculated correctly."""
        mock_func = Mock(side_effect=[ValueError("1"), ValueError("2"), ValueError("3"), "success"])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep") as mock_sleep,
            patch("ray_curator.utils.misc.retry_utils.logger"),
        ):
            result = do_with_retries(
                mock_func,
                backoff_factor=2.5,
                max_wait_time_s=10.0
            )

            assert result == "success"
            # Expected: 2.5^1=2.5, 2.5^2=6.25, 2.5^3=15.625 capped at 10.0
            expected_sleeps = [2.5, 6.25, 10.0]
            actual_sleeps = [call[0][0] for call in mock_sleep.call_args_list]
            assert actual_sleeps == expected_sleeps

    def test_exception_inheritance(self) -> None:
        """Test that exception inheritance works correctly for retry logic."""
        class CustomError(ValueError):
            pass

        mock_func = Mock(side_effect=[CustomError("custom error"), "success"])

        with (
            patch("ray_curator.utils.misc.retry_utils.time.sleep"),
            patch("ray_curator.utils.misc.retry_utils.logger"),
        ):
            # Should retry CustomError because it inherits from ValueError
            result = do_with_retries(mock_func, exceptions_to_retry=[ValueError])

            assert result == "success"
            assert mock_func.call_count == 2

    def test_actual_sleep_called(self) -> None:
        """Test that time.sleep is actually called with correct duration."""
        mock_func = Mock(side_effect=[ValueError("fail"), "success"])

        with patch("ray_curator.utils.misc.retry_utils.logger"):
            start_time = time.time()
            result = do_with_retries(mock_func, backoff_factor=0.1)  # Small backoff for quick test
            end_time = time.time()

            assert result == "success"
            # Should have taken at least 0.1 seconds due to sleep
            assert end_time - start_time >= 0.1
