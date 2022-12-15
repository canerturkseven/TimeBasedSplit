#%%
import pandas as pd
from TimeBasedSplit import TimeBasedSplit

df_day = pd.DataFrame(
    {
        "date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
    }
)


def test_daily():
    model = TimeBasedSplit(
        date_frequency="days",
        date_col="date",
        n_splits=2,
        forecast_horizon=[1, 3],
        max_train_size=3,
        end_offset=2,
        step_length=3,
    )
    correct_result = [
        ([2, 3, 4], [5, 7]),
        ([0, 1], [2, 4]),
    ]
    model_result = model.split(df_day)
    for cv_train, cv_test in model_result:
        print(
            pd.concat(
                [
                    df_day.loc[df_day.index.isin(cv_train)].assign(status="train"),
                    df_day.loc[~df_day.index.isin(cv_train + cv_test)].assign(
                        status=""
                    ),
                    df_day.loc[df_day.index.isin(cv_test)].assign(status="test"),
                ]
            ).sort_values(by="date")
        )
    assert correct_result == model_result


# %%
