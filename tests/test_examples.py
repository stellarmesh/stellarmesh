from glob import glob
from itertools import chain, count
from pathlib import Path

import pytest


def find_examples(pattern="examples/*.py", path=Path("examples")):
    for p in glob(pattern):
        with open(p, encoding="UTF-8") as f:
            code = f.read()

        yield code, path


@pytest.mark.parametrize("code, path", find_examples(), ids=count(0))
def test_example(code, path):
    exec(code)
