import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class TimeBasedSplit:
    def __init__(
        self,
        *,
        date_frequency,
        test_size=1,
        max_train_size=None,
        gap=0,
        n_splits=5
    ):
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.n_splits = n_splits
        self.date_frequency = date_frequency
        self.gap = gap

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        if int(value) <= 0:
            raise ValueError('test_size must be positive')
        else:
            self._test_size = value

    @property
    def max_train_size(self):
        return self._max_train_size

    @max_train_size.setter
    def max_train_size(self, value):
        if value:
            if int(value) <= 0:
                raise ValueError('max_train_size must be positive')
        self._max_train_size = value

    @property
    def gap(self):
        return self._gap

    @gap.setter
    def gap(self, value):
        if int(value) < 0:
            raise ValueError('gap must be greater than zero')
        else:
            self._gap = value

    @property
    def n_splits(self):
        return self._n_splits

    @n_splits.setter
    def n_splits(self, value):
        if int(value) <= 0:
            raise ValueError('n_splits must be positive')
        else:
            self._n_splits = value

    @property
    def date_frequency(self):
        return self._date_frequency

    @date_frequency.setter
    def date_frequency(self, value):
        supported_date_frequency = [
            'years', 'months', 'weeks', 'days',
            'hours', 'minutes', 'seconds', 'microseconds'
        ]
        if not value in supported_date_frequency:
            raise ValueError(f'{value} is not supported as date frequency')
        else:
            self._date_frequency = value

    def check_input(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df must be a pandas DataFrame')

    def check_date_col(self, df, date_col):
        if not date_col in df.columns:
            raise ValueError(f'{date_col} is not in the {df} columns')
        if not np.issubdtype(df[date_col].dtypes, np.datetime64):
            raise ValueError(f'{date_col} must be a date column')

    def check_n_splits(self, df, date_col):
        max_date = df[date_col].max().to_pydatetime()
        min_date = (
            max_date
            - relativedelta(**{self.date_frequency: self.gap})
            - (relativedelta(**{self.date_frequency: self.test_size})
               * self.n_splits)
        )
        if df[df[date_col] <= min_date].empty:
            raise ValueError(
                f"Too many splits={self.n_splits} "
                f"with test_size={self.test_size} and gap={self.gap} "
                f"for the date sequence."
            )

    def split(self, df, date_col):
        self.check_input(df)
        self.check_date_col(df, date_col)
        self.check_n_splits(df, date_col)
        gap = self.gap
        max_train_size = self.max_train_size
        date_frequency = self.date_frequency
        test_size = self.test_size
        n_splits = self.n_splits
        #Reset df index just in case
        df = df.reset_index(drop=True)
        max_date = df[date_col].max().to_pydatetime()
        splits = []
        for i in range(n_splits):
            test_end = (
                max_date - i*relativedelta(**{date_frequency: test_size})
            )
            test_start = (
                max_date - (i+1) * relativedelta(**{date_frequency: test_size})
            )
            train_end = (
                test_start - relativedelta(**{date_frequency: gap})
            )
            test_condition = (
                (df[date_col] > test_start) & (df[date_col] <= test_end)
            )
            if self.max_train_size:
                train_start = (
                    train_end -
                    relativedelta(**{date_frequency: max_train_size})
                )
                train_condition = (
                    (df[date_col] > train_start) & (df[date_col] <= train_end)
                )
            else:
                train_condition = (df[date_col] <= train_end)
            splits.append(
                (
                    df[train_condition].index.tolist(),
                    df[test_condition].index.tolist()
                )
            )
        return splits
