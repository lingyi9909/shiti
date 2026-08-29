from question_builder.parser.docx.numbering import NumberingResolver

NUMBERING_XML = """<?xml version='1.0' encoding='UTF-8'?>
<w:numbering xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>
  <w:abstractNum w:abstractNumId='0'>
    <w:lvl w:ilvl='0'>
      <w:start w:val='1'/><w:numFmt w:val='decimal'/><w:lvlText w:val='%1.'/>
    </w:lvl>
    <w:lvl w:ilvl='1'>
      <w:start w:val='1'/><w:numFmt w:val='decimal'/><w:lvlText w:val='（%2）'/>
    </w:lvl>
  </w:abstractNum>
  <w:abstractNum w:abstractNumId='1'>
    <w:lvl w:ilvl='0'>
      <w:start w:val='1'/><w:numFmt w:val='upperLetter'/><w:lvlText w:val='%1.'/>
    </w:lvl>
  </w:abstractNum>
  <w:abstractNum w:abstractNumId='2'>
    <w:lvl w:ilvl='0'>
      <w:start w:val='1'/><w:numFmt w:val='chineseCountingThousand'/>
      <w:lvlText w:val='%1、'/>
    </w:lvl>
  </w:abstractNum>
  <w:abstractNum w:abstractNumId='3'>
    <w:lvl w:ilvl='0'>
      <w:start w:val='1'/><w:numFmt w:val='decimalEnclosedCircle'/>
      <w:lvlText w:val='%1'/>
    </w:lvl>
  </w:abstractNum>
  <w:num w:numId='10'><w:abstractNumId w:val='0'/></w:num>
  <w:num w:numId='11'><w:abstractNumId w:val='1'/></w:num>
  <w:num w:numId='12'><w:abstractNumId w:val='2'/></w:num>
  <w:num w:numId='13'><w:abstractNumId w:val='3'/></w:num>
  <w:num w:numId='14'>
    <w:abstractNumId w:val='0'/>
    <w:lvlOverride w:ilvl='0'><w:startOverride w:val='5'/></w:lvlOverride>
  </w:num>
  <w:num w:numId='15'>
    <w:abstractNumId w:val='0'/>
    <w:lvlOverride w:ilvl='0'>
      <w:startOverride w:val='3'/>
      <w:lvl w:ilvl='0'>
        <w:start w:val='1'/><w:numFmt w:val='upperLetter'/><w:lvlText w:val='%1)'/>
      </w:lvl>
    </w:lvlOverride>
  </w:num>
</w:numbering>
""".encode()


def test_resolves_decimal_and_multilevel_labels() -> None:
    resolver = NumberingResolver.from_xml(NUMBERING_XML)

    assert resolver.next_label(10, 0).resolved_label == "1."
    assert resolver.next_label(10, 1).resolved_label == "（1）"
    assert resolver.next_label(10, 1).resolved_label == "（2）"
    assert resolver.next_label(10, 0).resolved_label == "2."


def test_resolves_letter_chinese_and_enclosed_circle_numbering() -> None:
    resolver = NumberingResolver.from_xml(NUMBERING_XML)

    assert resolver.next_label(11, 0).resolved_label == "A."
    assert resolver.next_label(11, 0).resolved_label == "B."
    assert resolver.next_label(12, 0).resolved_label == "一、"
    assert resolver.next_label(12, 0).resolved_label == "二、"
    assert resolver.next_label(13, 0).resolved_label == "①"
    assert resolver.next_label(13, 0).resolved_label == "②"


def test_num_instance_start_override_restarts_from_custom_value() -> None:
    resolver = NumberingResolver.from_xml(NUMBERING_XML)

    assert resolver.next_label(14, 0).resolved_label == "5."
    assert resolver.next_label(14, 0).resolved_label == "6."


def test_level_override_replaces_format_and_honors_start_override() -> None:
    resolver = NumberingResolver.from_xml(NUMBERING_XML)

    assert resolver.next_label(15, 0).resolved_label == "C)"
    assert resolver.next_label(15, 0).resolved_label == "D)"


def test_resolution_preserves_numbering_evidence() -> None:
    resolver = NumberingResolver.from_xml(NUMBERING_XML)

    resolved = resolver.next_label(10, 1)

    assert resolved.num_id == 10
    assert resolved.abstract_num_id == 0
    assert resolved.ilvl == 1
    assert resolved.lvl_text == "（%2）"
    assert resolved.start == 1
    assert resolved.value == 1
