from question_builder.parser.docx.numbering import NumberingResolver

NUMBERING_XML = """<?xml version='1.0' encoding='UTF-8'?>
<w:numbering xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>
  <w:abstractNum w:abstractNumId='0'>
    <w:lvl w:ilvl='0'><w:start w:val='1'/><w:numFmt w:val='decimal'/><w:lvlText w:val='%1.'/></w:lvl>
    <w:lvl w:ilvl='1'><w:start w:val='1'/><w:numFmt w:val='decimal'/><w:lvlText w:val='（%2）'/></w:lvl>
  </w:abstractNum>
  <w:abstractNum w:abstractNumId='1'>
    <w:lvl w:ilvl='0'><w:start w:val='1'/><w:numFmt w:val='upperLetter'/><w:lvlText w:val='%1.'/></w:lvl>
  </w:abstractNum>
  <w:abstractNum w:abstractNumId='2'>
    <w:lvl w:ilvl='0'><w:start w:val='1'/><w:numFmt w:val='chineseCountingThousand'/><w:lvlText w:val='%1、'/></w:lvl>
  </w:abstractNum>
  <w:num w:numId='10'><w:abstractNumId w:val='0'/></w:num>
  <w:num w:numId='11'><w:abstractNumId w:val='1'/></w:num>
  <w:num w:numId='12'><w:abstractNumId w:val='2'/></w:num>
</w:numbering>
""".encode()


def test_resolves_decimal_and_multilevel_labels() -> None:
    resolver = NumberingResolver.from_xml(NUMBERING_XML)

    assert resolver.next_label(10, 0).resolved_label == "1."
    assert resolver.next_label(10, 1).resolved_label == "（1）"
    assert resolver.next_label(10, 1).resolved_label == "（2）"
    assert resolver.next_label(10, 0).resolved_label == "2."


def test_resolves_letter_and_chinese_numbering() -> None:
    resolver = NumberingResolver.from_xml(NUMBERING_XML)

    assert resolver.next_label(11, 0).resolved_label == "A."
    assert resolver.next_label(11, 0).resolved_label == "B."
    assert resolver.next_label(12, 0).resolved_label == "一、"
    assert resolver.next_label(12, 0).resolved_label == "二、"


def test_resolution_preserves_numbering_evidence() -> None:
    resolver = NumberingResolver.from_xml(NUMBERING_XML)

    resolved = resolver.next_label(10, 1)

    assert resolved.num_id == 10
    assert resolved.abstract_num_id == 0
    assert resolved.ilvl == 1
    assert resolved.lvl_text == "（%2）"
    assert resolved.start == 1
    assert resolved.value == 1
