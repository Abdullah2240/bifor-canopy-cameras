import numpy as np
import pandas as pd
import pytest

from bifor import timeseries as ts


def test_hampel_flags_spike_only():
    s = pd.Series(np.linspace(0.2, 0.5, 30))
    s.iloc[12] = 2.0
    out = ts.hampel(s)
    assert out.tolist().count(True) == 1 and out.iloc[12]


def test_logistic_fit_recovers_greenup_dates():
    rng = np.random.default_rng(3)
    doy = np.arange(-20, 160, 1.0)
    truth = ts.logistic(doy, 0.15, 0.6, 125.0, 0.12)
    y = truth + rng.normal(0, 0.02, doy.size)
    fit = ts.fit_greenup(doy, y, np.ones_like(doy), n_boot=100, seed=0)
    assert fit["mid50"] == pytest.approx(125, abs=2)
    assert fit["sos20"] < fit["mid50"] < fit["mat80"]
    assert fit["mid50_lo"] < 125 < fit["mid50_hi"]


def test_daily_uses_only_qa_pass_frames_near_noon():
    t = pd.to_datetime(["2026-05-01 12:00", "2026-05-01 12:30", "2026-05-01 06:00", "2026-05-01 13:00"], utc=True)
    ind = pd.DataFrame({"frame_id": list("abcd"), "camera": "A5", "roi": "canopy_lower", "index": "NDVI",
                        "value": [0.5, 0.6, 0.0, 9.0]})
    frames = pd.DataFrame({"frame_id": list("abcd"), "ts_utc": t, "min_from_noon": [-10, 20, -370, 50],
                           "qa_pass": [True, True, True, False]})
    d = ts.daily(ind, frames, window_min=120)
    assert len(d) == 1
    row = d.iloc[0]
    assert row.n_frames == 2 and row["median"] == pytest.approx(0.55)


def test_power_of_logistic_shift_detection_is_monotone():
    p = [ts.shift_detect_power(shift, n_sim=40, seed=1) for shift in (0, 10)]
    assert p[0] < 0.2 < p[1]


def test_full_season_uses_double_logistic_and_recovers_both_dates():
    rng = np.random.default_rng(4)
    doy = np.arange(60, 330, 1.0)
    truth = ts.double_logistic(doy, 0.2, 0.6, 125.0, 0.12, 285.0, 0.1)
    y = truth + rng.normal(0, 0.02, doy.size)
    fit = ts.fit_season(doy, y, np.ones_like(doy), n_boot=60, seed=0)
    assert fit["model"] == "double_logistic"
    assert fit["mid50"] == pytest.approx(125, abs=3)
    assert fit["eos50"] == pytest.approx(285, abs=3)
    assert fit["eos50_lo"] < 285 < fit["eos50_hi"]


def test_spring_only_record_uses_single_logistic():
    rng = np.random.default_rng(5)
    doy = np.arange(-20, 154, 1.0)
    y = ts.logistic(doy, 0.2, 0.5, 125.0, 0.15) + rng.normal(0, 0.02, doy.size)
    fit = ts.fit_season(doy, y, np.ones_like(doy), n_boot=40, seed=0)
    assert fit["model"] == "logistic" and np.isnan(fit["eos50"])


def test_half_rise_date_finds_midpoint_of_observed_rise():
    import pandas as pd
    dates = pd.date_range("2026-03-10", "2026-06-03", freq="D")
    t = (dates - pd.Timestamp("2026-01-01")).days.to_numpy(float)
    y = ts.logistic(t, 0.2, 0.6, 125.0, 0.3)          # midpoint 6 May (doy 125)
    d = ts.half_rise_date(pd.Series(y, index=dates), base_end="2026-04-20")
    assert abs((d - pd.Timestamp("2026-05-06")).days) <= 1


def test_paired_shift_recovers_known_shift_with_shared_noise():
    import numpy as np
    rng = np.random.default_rng(1)
    t = np.arange(68.0, 155.0)
    common = rng.normal(0, 0.03, t.size)
    ya = ts.logistic(t, 0.2, 0.6, 128.0, 0.12) + common + rng.normal(0, 0.01, t.size)
    yb = ts.logistic(t, 0.2, 0.6, 122.0, 0.12) + common + rng.normal(0, 0.01, t.size)
    out = ts.paired_shift(t, ya, yb, n_boot=200)
    assert abs(out["shift"] - 6.0) < 2.0                       # a minus b, days
    assert out["lo"] < out["shift"] < out["hi"]
    assert out["hi"] - out["lo"] < 8.0


def test_half_rise_date_ignores_brief_crossing_before_a_pause():
    dates = pd.date_range("2026-03-10", "2026-06-03", freq="D")
    y = pd.Series(0.0, index=dates)
    y[dates >= "2026-05-01"] = 0.55          # brief step above half
    y[dates >= "2026-05-08"] = 0.40          # pause back below
    y[dates >= "2026-05-20"] = 1.0           # final rise
    d = ts.half_rise_date(y, base_end="2026-04-20", smooth=1)
    assert d == pd.Timestamp("2026-05-20")
