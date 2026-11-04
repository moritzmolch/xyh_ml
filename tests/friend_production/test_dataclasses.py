import dataclasses

import pytest

from xyh_ml.config.dataclasses import Campaign, Friend
from xyh_ml.friend_production.dataclasses import Sample, Shift, Variable


def test_variable_stores_fields():
    v = Variable(name="pt", column="pt__jesUp")
    assert (v.name, v.column) == ("pt", "pt__jesUp")


def test_shift_stores_fields():
    v = Variable(name="pt", column="pt__jesUp")
    s = Shift(name="jesUp", variables=[v])
    assert s.name == "jesUp"
    assert s.variables == [v]


def test_sample_roundtrips_through_asdict():
    s = Sample(
        era="2018",
        sample="A",
        channel="et",
        main_files=["m.root"],
        friend_files={"f": ["f.root"]},
        shifts=[Shift(name="jesUp", variables=[Variable("pt", "c")])],
    )
    d = dataclasses.asdict(s)
    assert d["shifts"] == [
        {"name": "jesUp", "variables": [{"name": "pt", "column": "c"}]}
    ]
    assert d["friend_files"] == {"f": ["f.root"]}


def test_sample_equality():
    kw = dict(era="e", sample="s", channel="c", main_files=[], friend_files={})
    assert Sample(shifts=[], **kw) == Sample(shifts=[], **kw)


def test_campaign_fields():
    c = Campaign(name="c22", era="2022")
    assert (c.name, c.era) == ("c22", "2022")


def test_friend_defaults_are_none():
    f = Friend(name="f")
    assert f.include_channels is None
    assert f.exclude_samples is None


@pytest.mark.parametrize("kind", ["channels", "campaigns", "samples"])
def test_friend_rejects_include_and_exclude(kind):
    kwargs = {f"include_{kind}": ["a"], f"exclude_{kind}": ["b"]}
    with pytest.raises(ValueError, match=f"both include and exclude {kind}"):
        Friend(name="f", **kwargs)


def test_friend_allows_include_and_exclude_of_different_kinds():
    f = Friend(name="f", include_channels=["et"], exclude_samples=["x"])
    assert f.include_channels == ["et"]


def test_sample_from_dict_roundtrip():
    s = Sample(
        era="2018",
        sample="A",
        channel="et",
        main_files=["m.root"],
        friend_files={"f": ["f.root"]},
        shifts=[Shift(name="jesUp", variables=[Variable("pt", "c")])],
    )
    restored = Sample.from_dict(dataclasses.asdict(s))
    assert restored == s
    assert isinstance(restored.shifts[0], Shift)
    assert isinstance(restored.shifts[0].variables[0], Variable)
