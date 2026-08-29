from adaptive.ingestion.models import ParsedDocument
from adaptive.ingestion.normalize import normalize

SHA256_HEX_LENGTH = 64


def test_normalize_canonicalizes_whitespace_markers_links_and_hash():
    parsed = ParsedDocument(
        title="  Handbook  ",
        text="# Handbook\r\n\r\n  First   paragraph.  \n\n\nSecond paragraph.\n[Policy](https://example.test/policy)",
        pages=[{"page_number": 1, "text": "First"}],
        headings=[{"level": 1, "text": "Handbook"}],
        metadata={"author": "Ada", "source_links": ["https://example.test/policy"]},
    )

    document = normalize(parsed, acl={"support"})

    assert document.title == "Handbook"
    assert document.canonical_text == (
        "# Handbook\n\nFirst paragraph.\n\nSecond paragraph.\n[Policy](https://example.test/policy)"
    )
    assert document.metadata["author"] == "Ada"
    assert document.metadata["source_links"] == ["https://example.test/policy"]
    assert document.acl == frozenset({"support"})
    assert len(document.content_hash) == SHA256_HEX_LENGTH


def test_normalize_preserves_page_heading_and_table_metadata():
    parsed = ParsedDocument(
        title=None,
        text="Text",
        pages=[{"page_number": 2, "text": "Text"}],
        headings=[{"level": 2, "text": "Section"}],
        tables=[{"page_number": 2, "rows": [["A", "B"]]}],
    )
    document = normalize(parsed)
    assert document.pages == parsed.pages
    assert document.headings == parsed.headings
    assert document.tables == parsed.tables


def test_normalize_uses_first_heading_as_fallback_title():
    parsed = ParsedDocument(
        title=None, text="Section\nBody", headings=[{"level": 1, "text": "Section"}]
    )
    assert normalize(parsed).title == "Section"
