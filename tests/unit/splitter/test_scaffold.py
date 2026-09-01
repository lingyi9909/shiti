from importlib.util import find_spec


def test_task5_splitter_and_markdown_modules_exist() -> None:
    assert find_spec("question_builder.splitter.rules") is not None
    assert find_spec("question_builder.splitter.builder") is not None
    assert find_spec("question_builder.export.markdown") is not None
