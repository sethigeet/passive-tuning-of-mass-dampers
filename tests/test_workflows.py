from tmd.io import load_record


def test_prepared_el_centro_record_loads():
    record = load_record("el_centro")
    assert record.dt > 0.0
    assert len(record.time) > 100


def test_prepared_far_field_record_loads():
    record = load_record("northridge")
    assert record.dt > 0.0
    assert len(record.time) > 100
