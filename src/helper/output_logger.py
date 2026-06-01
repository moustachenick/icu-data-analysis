import os
import sys
from contextlib import contextmanager
from datetime import datetime


class _Tee:
    """A writable stream that forwards everything to several underlying streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except UnicodeEncodeError:
                # The console may use a narrow codepage (e.g. cp1252) that can't
                # represent the box-drawing characters tabulate emits. Degrade that
                # stream gracefully; the UTF-8 log file still receives the exact text.
                encoding = getattr(stream, "encoding", None) or "utf-8"
                stream.write(data.encode(encoding, errors="replace").decode(encoding))
            # Flush so the console stays responsive and the file mirrors it live.
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        # Defer to the real console so libraries that probe for a TTY behave.
        return getattr(self.streams[0], "isatty", lambda: False)()


def build_output_path(mode, output_dir="output"):
    """
    Build a timestamped output path, e.g. ``output/regression_20260601-143012.txt``.

    A fresh timestamped file per run preserves the full history, so any two runs can
    be diffed against each other as the code changes across refactoring steps.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(output_dir, f"{mode}_{timestamp}.txt")


@contextmanager
def tee_output(file_path):
    """
    Mirror everything written to stdout into ``file_path`` as well as the console.

    All the pipelines communicate through ``print`` (including ``df.info()`` and the
    tabulated tables), so teeing stdout captures the full run with no changes to the
    individual modules.

    Usage:
        with tee_output(build_output_path("regression")):
            main(...)
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    original_stdout = sys.stdout
    with open(file_path, "w", encoding="utf-8") as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        try:
            yield file_path
        finally:
            sys.stdout = original_stdout
