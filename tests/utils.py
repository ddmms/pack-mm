"""Utility functions for tests."""

# lifted from janus_core github.com/stfc/janus-core/
from __future__ import annotations

import re


def strip_ansi_codes(output: str) -> str:
    """
    Remove any ANSI sequences from output string.

    Based on:
    https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python/14693789#14693789

    Parameters
    ----------
    output
        Output that may contain ANSI sequences to be removed.

    Returns
    -------
    str
        Output with ANSI sequences removed.
    """
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(
        r"""
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    """,
        re.VERBOSE,
    )
    return ansi_escape.sub("", output)
