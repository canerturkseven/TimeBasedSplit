#%%
import pandas as pd
from TimeBasedSplit import TimeBasedSplit

pd.set_option("display.max_rows", 100)
df = pd.DataFrame(
    {
        "date": pd.date_range(start="2022-01-01", periods=100, freq="D"),
    }
)


def test_daily():
    model = TimeBasedSplit(
        date_frequency="days",
        test_size=1,
        max_train_size=None,
        gap=0,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(99)), list(range(99, 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_daily_test_size():
    model = TimeBasedSplit(
        date_frequency="days",
        test_size=3,
        max_train_size=None,
        gap=0,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(97)), list(range(97, 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_daily_gap():
    model = TimeBasedSplit(
        date_frequency="days",
        test_size=1,
        max_train_size=None,
        gap=1,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(98)), list(range(99, 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_daily_n_splits():
    model = TimeBasedSplit(
        date_frequency="days",
        test_size=1,
        max_train_size=None,
        gap=0,
        n_splits=2,
        date_col="date",
    )
    correct_result = [
        (list(range(99)), list(range(99, 100))),
        (list(range(98)), list(range(98, 99))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_daily_max_train_size():
    model = TimeBasedSplit(
        date_frequency="days",
        test_size=1,
        max_train_size=5,
        gap=0,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(94, 99)), list(range(99, 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_daily_combined():
    model = TimeBasedSplit(
        date_frequency="days",
        test_size=2,
        max_train_size=4,
        gap=1,
        n_splits=3,
        date_col="date",
    )
    correct_result = [
        (list(range(93, 97)), list(range(98, 100))),
        (list(range(91, 95)), list(range(96, 98))),
        (list(range(89, 93)), list(range(94, 96))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_weekly():
    model = TimeBasedSplit(
        date_frequency="weeks",
        test_size=1,
        max_train_size=None,
        gap=0,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (7 * 1))), list(range(100 - (7 * 1), 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_weekly_test_size():
    model = TimeBasedSplit(
        date_frequency="weeks",
        test_size=3,
        max_train_size=None,
        gap=0,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (3 * 7))), list(range(100 - (3 * 7), 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_weekly_gap():
    model = TimeBasedSplit(
        date_frequency="weeks",
        test_size=1,
        max_train_size=None,
        gap=1,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (2 * 7))), list(range(100 - (1 * 7), 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_weekly_n_splits():
    model = TimeBasedSplit(
        date_frequency="weeks",
        test_size=1,
        max_train_size=None,
        gap=0,
        n_splits=2,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (1 * 7))), list(range(100 - (1 * 7), 100))),
        (list(range(100 - (2 * 7))), list(range(100 - (2 * 7), 100 - (1 * 7)))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_weekly_max_train_size():
    model = TimeBasedSplit(
        date_frequency="weeks",
        test_size=1,
        max_train_size=5,
        gap=0,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (6 * 7), 100 - (1 * 7))), list(range(100 - (1 * 7), 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_weekly_combined():
    model = TimeBasedSplit(
        date_frequency="weeks",
        test_size=2,
        max_train_size=4,
        gap=1,
        n_splits=3,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (7 * 7), 100 - (3 * 7))), list(range(100 - (2 * 7), 100))),
        (
            list(range(100 - (9 * 7), 100 - (5 * 7))),
            list(range(100 - (4 * 7), 100 - (2 * 7))),
        ),
        (
            list(range(100 - (11 * 7), 100 - (7 * 7))),
            list(range(100 - (6 * 7), 100 - (4 * 7))),
        ),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_monthly():
    model = TimeBasedSplit(
        date_frequency="months",
        test_size=1,
        max_train_size=None,
        gap=0,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - 31)), list(range(100 - 31, 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_monthly_test_size():
    model = TimeBasedSplit(
        date_frequency="months",
        test_size=2,
        max_train_size=None,
        gap=0,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (28 + 31))), list(range(100 - (28 + 31), 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_monthly_gap():
    model = TimeBasedSplit(
        date_frequency="months",
        test_size=1,
        max_train_size=None,
        gap=1,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (31 + 28))), list(range(100 - (31), 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_monthly_n_splits():
    model = TimeBasedSplit(
        date_frequency="months",
        test_size=1,
        max_train_size=None,
        gap=0,
        n_splits=2,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (31))), list(range(100 - (31), 100))),
        (list(range(100 - (31 + 28))), list(range(100 - (31 + 28), 100 - (31)))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_monthly_max_train_size():
    model = TimeBasedSplit(
        date_frequency="months",
        test_size=1,
        max_train_size=2,
        gap=0,
        n_splits=1,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (31 + 28 + 31), 100 - (31))), list(range(100 - (31), 100))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


def test_monthly_combined():
    model = TimeBasedSplit(
        date_frequency="months",
        test_size=1,
        max_train_size=2,
        gap=1,
        n_splits=2,
        date_col="date",
    )
    correct_result = [
        (list(range(100 - (31 + 28))), list(range(100 - (31), 100))),
        (list(range(100 - (31 + 28 + 31))), list(range(100 - (31 + 28), 100 - (31)))),
    ]
    model_result = model.split(df)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df.loc[df.index.isin(cv_train)].assign(status="train"),
                    df.loc[~df.index.isin(cv_train + cv_test)].assign(status=""),
                    df.loc[df.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


# %%
