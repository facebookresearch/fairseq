# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from datetime import date, datetime
from typing import List, Optional, Tuple

from fairseq.data import FairseqDataset, FairseqIterableDataset


KOSKI_THREADS = 16

logger = logging.getLogger(__name__)


def _set_up_dataframe(
    table,
    namespace,
    column_projections: List[str] = None,
    where_clause: str = None,
    limit: int = None,
    shuffle: bool = False,
):
    import koski.dataframes as kd

    # Silencing Koski logs
    kd.set_min_log_level(2)

    ctx = kd.create_ctx(
        use_case=kd.UseCase.PROD,
        description="streaming data into fairseq models",
        oncall="fairseq",
    )
    dataframe = kd.data_warehouse(
        namespace=namespace,
        table=table,
        session_ctx=ctx,
    )
    if column_projections:
        dataframe = dataframe.map(column_projections)
    if where_clause:
        dataframe = dataframe.filter(where_clause)
    if limit is not None:
        dataframe = dataframe.limit(limit)
    if shuffle:
        dataframe = dataframe.shuffle(memory_limit=int(12 * 1024 * 1024))  # 12GB
    return dataframe


def _date_from_string(date_string: str) -> date:
    return datetime.strptime(date_string, "%Y-%m-%d").date()


def _date_where_clause(date_ranges) -> Optional[str]:
    if not date_ranges:
        return None
    clauses = []
    for ds_range in date_ranges:
        # sanitize
        old_date = _date_from_string(ds_range[0]).isoformat()
        new_date = _date_from_string(ds_range[1]).isoformat()
        clauses.append(f"(ds >= '{old_date}' AND ds <= '{new_date}')")
    return f"({' OR '.join(clauses)})" if clauses else None


class HiveDataset(FairseqDataset):
    """
    Used to read data from a Hive table. Loads all data into memory on
    instantiation.

    Given a query, this will returns tuples, like:
        [('col1 val1', 'col2 val1'), ('col1 val2', 'col2 val2'), ...]

    Args:
        table: Data warehouse table to query from.
        namespace: Data warehouse namespace in which that table lives.
        date_ranges: List of tuples of date ranges from which to fetch data,
            each in yyyy-mm-dd format. Example: [(2019-12-31, 2020-01-01)]
        limit: Limit on the total number of rows to fetch.
        filter_fn: A function that takes in a row and outputs a bool. Can be
            used to filter data at query time to save memory.
    """

    def __init__(
        self,
        table: str,
        namespace: str,
        date_ranges: List[Tuple[str, str]],
        limit=None,
        filter_fn=None,
    ) -> None:
        super().__init__()
        dataframe = _set_up_dataframe(
            table=table,
            namespace=namespace,
            where_clause=_date_where_clause(date_ranges),
            limit=limit,
        )
        logger.info("Loading Hive data...")
        self.data = []
        for c in dataframe.rows(num_worker_threads=KOSKI_THREADS):
            if filter_fn is not None and not filter_fn(c):
                continue
            self.data.append(c)
        logger.info(f"Finished loading {len(self.data)} rows")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for c in self.data:
            yield c

    def __getitem__(self, index):
        return self.data[index]


class StreamingHiveDataset(FairseqIterableDataset):
    """Used to stream data from a Hive table.

    Given a query, this will returns tuples, like:
        [('col1 val1', 'col2 val1'), ('col1 val2', 'col2 val2'), ...]

    Args:
        table: Hive table to query from.
        namespace: Data warehouse namespace in which that table lives.
        limit: Limit on the total number of rows to fetch.
        where_clause: SQL filter to be appended (via 'AND') to the query.
            Note that only Koski functions are supported.
        date_ranges: List of tuples of date ranges from which to fetch data,
            each in yyyy-mm-dd format. Example: [(2019-12-31, 2020-01-01)]
        shuffle: Performs a total shuffle across data taken from date_ranges.
            Note that fresh_date_ranges is not shuffled.
        fresh_date_ranges: Date ranges considered 'fresh' will be sampled at a
            constant ratio with the rest of the data. To ensure the ratio is
            accurate, these dates should not overlap with date_ranges.
        fresh_ratio: Ratio of date_ranges to fresh_date_ranges. Must be a
            positive integer. For example, if 1/4 of the data should come from
            fresh_date_ranges, fresh_ratio should be 4.
    """

    def __init__(
        self,
        table: str,
        namespace: str,
        limit: int,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        date_ranges: Optional[List[Tuple[str, str]]] = None,
        shuffle=False,
        shuffle_col: str = "thread_key",
        fresh_date_ranges: Optional[List[Tuple[str, str]]] = None,
        fresh_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.table = table
        self.namespace = namespace
        self.limit = limit

        self.columns = columns
        self.given_filter = where_clause
        self.date_ranges = date_ranges
        self.fresh_date_ranges = fresh_date_ranges
        self.fresh_ratio = fresh_ratio
        self.shuffle = shuffle
        self.shuffle_col = shuffle_col

    def __len__(self):
        return self.limit

    def __iter__(self):
        iterable = None
        if self.shuffle:
            iterable = self._shuffled_iterable()
        else:
            iterable = self._ordered_iterable()

        fresh_iterable = None
        if self.fresh_date_ranges:
            fresh_iterable = self._fresh_iterable()

        row_count = 0
        while row_count < self.limit:
            if fresh_iterable is not None and row_count % self.fresh_ratio == 0:
                yield next(fresh_iterable)
            else:
                yield next(iterable)
            row_count += 1

    def _fresh_iterable(self):
        dataframe = _set_up_dataframe(
            table=self.table,
            namespace=self.namespace,
            column_projections=self.columns,
            where_clause=self._build_where_clause(
                date_clause=_date_where_clause(self.fresh_date_ranges),
            ),
            limit=self.limit,
        )
        for c in dataframe.rows(num_worker_threads=KOSKI_THREADS):
            yield c

    def _shuffled_iterable(self):
        # Run through training examples in random slices to shuffle
        num_slices = 100
        slices = [i for i in range(num_slices)]
        random.shuffle(slices)
        for i in slices:
            dataframe = _set_up_dataframe(
                table=self.table,
                namespace=self.namespace,
                column_projections=self.columns,
                where_clause=self._build_where_clause(
                    date_clause=_date_where_clause(self.date_ranges),
                    shuffle_clause=f"abs(hash({self.shuffle_col}) % {num_slices}) = {i}",
                ),
                limit=self.limit,
            )
            for c in dataframe.rows(num_worker_threads=KOSKI_THREADS):
                yield c

    def _ordered_iterable(self):
        dataframe = _set_up_dataframe(
            table=self.table,
            namespace=self.namespace,
            column_projections=self.columns,
            where_clause=self._build_where_clause(
                date_clause=_date_where_clause(self.date_ranges)
            ),
            limit=self.limit,
        )
        for c in dataframe.rows(num_worker_threads=KOSKI_THREADS):
            yield c

    def _build_where_clause(self, date_clause=None, shuffle_clause=None) -> str:
        clauses = [
            self.given_filter,
            date_clause,
            shuffle_clause,
        ]
        return " AND ".join([f"({x})" for x in clauses if x])
