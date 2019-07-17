# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import getpass
import logging

from fairseq.data import FairseqDataset

logger = logging.getLogger(__name__)
_USER = getpass.getuser()


def _set_up_dataframe(table, namespace, where_clause=None, limit=None):
    import koski.dataframes as kd

    ctx = kd.create_ctx(
        use_case=kd.UseCase.PROD,
        description='streaming data into fairseq models',
        oncall='fairseq',
    )
    dataframe = kd.data_warehouse(
        namespace=namespace,
        table=table,
        session_ctx=ctx,
    )
    if where_clause is not None:
        dataframe = dataframe.filter(where_clause)
    if limit is not None:
        dataframe = dataframe.limit(limit)
    return dataframe


class HiveDataset(FairseqDataset):
    """Used to stream data from a Hive table.

    Given a query, this will returns tuples, like:
        [('col1 val1', 'col2 val1'), ('col1 val2', 'col2 val2'), ...]

    Args:
        table (String): data warehouse table to query from
        namespace (String): data warehouse namespace in which that table lives
        limit (String, optional): limit on total number of rows to fetch
        where_clause (String, optional): SQL syntax; filter clause that would normally be placed after 'WHERE'
    """

    # Overridden functions from FairseqDataset

    def __init__(self, table, namespace, where_clause=None, limit=None):
        super().__init__()
        self.limit = limit
        self.dataframe = _set_up_dataframe(
            table=table,
            namespace=namespace,
            where_clause=where_clause,
            limit=limit,
        )

    def __getitem__(self, index):
        return self.results_iterator[index]

    def __len__(self):
        # max(limit, total number of rows)
        return self.limit

    def __iter__(self):
        for c in self.dataframe.rows():
            yield c

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        raise NotImplementedError

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise NotImplementedError
