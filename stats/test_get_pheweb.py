import gzip
from io import BytesIO

from stats.get_pheweb import aggregate_outputs, _reader_from_response


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.raw = BytesIO(payload)
        self.headers = {}


def test_reader_from_response_uses_sniffed_gzip_when_headers_missing():
    uncompressed = b"col1\tcol2\n1\t2\n"
    compressed = gzip.compress(uncompressed)
    resp = _FakeResponse(compressed)

    reader = _reader_from_response(resp, info={}, compression_info={"is_gzip": True})

    assert reader.read() == uncompressed


def test_aggregate_outputs_writes_placeholder_when_empty(tmp_path):
    output = tmp_path / "aggregated.tsv"

    aggregate_outputs(tmp_path, output)

    assert output.exists()
    assert output.read_text() == ""
