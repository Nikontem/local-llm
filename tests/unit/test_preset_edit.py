import pytest

from local_llm.preset import Preset, PresetError

BASE = "version = 1\n\n[*]\njinja = true\n\n[a]\nmodel = /m/a.gguf\nc = 4096\n"


def test_add_section_appends_with_comments_and_blank_line():
    p = Preset.parse(BASE)
    p.add_section("b", [("model", "/m/b.gguf"), ("c", "8192")], comments=["added by local-llm"])
    assert p.dump() == BASE + "\n# added by local-llm\n[b]\nmodel = /m/b.gguf\nc = 8192\n"
    assert p.sections() == ["a", "b"]
    assert p.get("b", "jinja") == "true"


def test_add_section_rejects_duplicates_and_bad_names():
    p = Preset.parse(BASE)
    with pytest.raises(PresetError, match="already exists"):
        p.add_section("a", [("model", "/x")])
    with pytest.raises(PresetError, match="Invalid section name"):
        p.add_section("bad]name", [("model", "/x")])


def test_replace_section_keeps_surroundings():
    text = BASE + "\n# old note\n[b]\nmodel = /m/b.gguf\nc = 1\n\n[c]\nmodel = /m/c.gguf\n"
    p = Preset.parse(text)
    p.replace_section("b", [("model", "/m/b2.gguf")], comments=["new note"])
    assert p.dump() == BASE + "\n# new note\n[b]\nmodel = /m/b2.gguf\n\n[c]\nmodel = /m/c.gguf\n"
    with pytest.raises(PresetError, match="No section"):
        p.replace_section("zzz", [])


def test_remove_section_takes_its_leading_comments():
    text = BASE + "\n# added by local-llm\n[b]\nmodel = /m/b.gguf\n\n[c]\nmodel = /m/c.gguf\n"
    p = Preset.parse(text)
    p.remove_section("b")
    assert p.dump() == BASE + "\n[c]\nmodel = /m/c.gguf\n"
    with pytest.raises(PresetError, match="No section"):
        p.remove_section("b")


def test_save_is_atomic_and_keeps_a_backup(tmp_path):
    target = tmp_path / "models.ini"
    target.write_text(BASE)
    p = Preset.load(target)
    p.add_section("b", [("model", "/m/b.gguf")])
    p.save(target)
    assert target.read_text().endswith("[b]\nmodel = /m/b.gguf\n")
    assert (tmp_path / "models.ini.bak").read_text() == BASE
    assert not list(tmp_path.glob("*.tmp"))


def test_save_creates_parent_directory(tmp_path):
    target = tmp_path / "deep" / "models.ini"
    p = Preset.parse("[*]\njinja = true\n")
    p.save(target)
    assert target.read_text() == "[*]\njinja = true\n"
