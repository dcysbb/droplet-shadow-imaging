import csv

from droplet_shadow.config import SimulationConfig
from droplet_shadow.sensitivity import detection_calibration
from droplet_shadow.simulation import simulate


def test_detection_calibration_uses_independent_held_out_exposures(tmp_path):
    config = SimulationConfig().with_updates(
        run={"engine": "ray", "n_simulated": 12, "exposure_electrons": 12})
    templates = {}
    for q in (0.0, 1e6):
        path = tmp_path / f"template_{q:.0f}"
        simulate(config.with_updates(charge={"q_e": q},
                                     run={"n_simulated": 30, "exposure_electrons": 12,
                                          "seed": 92821}), path)
        templates[q] = path
    result = detection_calibration(config, templates, tmp_path / "calibration",
                                   repetitions=4, fit_nuisance=False)
    assert result.calibration_repetitions == 2
    assert result.validation_repetitions == 2
    assert set(result.interval_coverage_by_true_q_e) == {0.0, 1e6}
    with (tmp_path / "calibration" / "calibration_scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["subset"] for row in rows} == {"calibration", "validation"}
