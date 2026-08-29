from pathlib import Path

import pytest

from adaptive.ingestion.models import SourcePayload
from adaptive.ingestion.parsers import (
    DocxParser,
    HtmlParser,
    MarkdownParser,
    PdfParser,
    PlainTextParser,
    parse_bytes,
    select_parser,
)
from adaptive.ingestion.sources import FilesystemSource, LocalUploadSource

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"


def test_parser_selection_supports_required_mime_types():
    assert isinstance(select_parser("application/pdf"), PdfParser)
    assert isinstance(
        select_parser("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        DocxParser,
    )
    assert isinstance(select_parser("text/html"), HtmlParser)
    assert isinstance(select_parser("text/markdown"), MarkdownParser)
    assert isinstance(select_parser("text/plain"), PlainTextParser)


def test_html_parser_removes_scripts_and_preserves_title():
    document = parse_bytes(
        b"<title>Policy</title><script>alert(1)</script><h1>Refunds</h1><p>Keep me</p>",
        "text/html",
    )
    assert document.title == "Policy"
    assert "alert(1)" not in document.text
    assert "Keep me" in document.text
    assert document.headings == [{"level": 1, "text": "Refunds"}]


def test_markdown_parser_preserves_headings_and_title():
    document = parse_bytes(FIXTURES.joinpath("sample.md").read_bytes(), "text/markdown")
    assert document.title == "Employee Handbook"
    assert {heading["text"] for heading in document.headings} == {
        "Employee Handbook",
        "Leave",
        "Security",
    }
    assert "Annual leave is tracked in days." in document.text


def test_plain_text_parser_normalizes_utf8_bom():
    document = parse_bytes(b"\xef\xbb\xbfCaf\xc3\xa9\r\nsecond", "text/plain")
    assert document.text == "Café\nsecond"


def test_pdf_and_docx_fixtures_preserve_source_locations():
    pdf = parse_bytes(FIXTURES.joinpath("sample.pdf").read_bytes(), "application/pdf")
    docx = parse_bytes(
        FIXTURES.joinpath("sample.docx").read_bytes(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert pdf.title == "Operations Guide"
    assert pdf.pages[0]["page_number"] == 1
    assert "Incident response" in pdf.text
    assert docx.title == "Operations Guide"
    assert docx.headings[1]["text"] == "Incident response"
    assert "Escalate within one hour." in docx.text


@pytest.mark.asyncio
async def test_local_upload_source_returns_payload_with_metadata():
    source = LocalUploadSource(b"hello", filename="note.txt", mime_type="text/plain")
    payload = await source.read()
    assert isinstance(payload, SourcePayload)
    assert payload.data == b"hello"
    assert payload.filename == "note.txt"
    assert payload.mime_type == "text/plain"
    assert payload.source_uri is None


def test_parse_payload_preserves_source_metadata():
    payload = SourcePayload(
        data=b"hello",
        filename="note.txt",
        mime_type="text/plain",
        source_uri="/allowed/note.txt",
        metadata={"origin": "upload"},
    )
    document = parse_bytes(payload)
    assert document.metadata == {
        "origin": "upload",
        "filename": "note.txt",
        "source_uri": "/allowed/note.txt",
    }


def test_sources_reject_oversized_upload_and_path_escape(tmp_path: Path):
    with pytest.raises(ValueError, match="size"):
        LocalUploadSource(b"1234", filename="x.txt", mime_type="text/plain", max_size=3)

    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="within"):
        FilesystemSource(outside, allowed_root=root)


@pytest.mark.asyncio
async def test_filesystem_source_reads_only_validated_file(tmp_path: Path):
    root = tmp_path / "allowed"
    root.mkdir()
    file_path = root / "guide.txt"
    file_path.write_text("guide", encoding="utf-8")
    payload = await FilesystemSource(file_path, allowed_root=root).read()
    assert payload.data == b"guide"
    assert payload.source_uri == str(file_path)
