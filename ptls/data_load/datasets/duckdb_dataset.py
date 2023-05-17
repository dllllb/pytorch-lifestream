from typing import Iterable, List
import warnings

import duckdb
import torch

from ptls.data_load import IterableChain


class DuckDbDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            data_read_func: str,
            col_id: str,
            col_event_time: str,
            col_event_fields: List[str],
            i_filters: List[Iterable] = None):
        self.data_read_func = data_read_func
        self.col_id = col_id
        self.col_event_time = col_event_time
        self.col_event_fields = col_event_fields

        if i_filters:
            self.post_processing = IterableChain(*i_filters)
        else:
            self.post_processing = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            pass
        else:  # in a worker process
            raise NotImplementedError('multiple worker processes are not supported yet')

        gen = self.__execute_query()
        if self.post_processing is not None:
            gen = self.post_processing(gen)
        return gen

    def __execute_query(self):
        event_fields = self.col_event_fields.copy() + [self.col_event_time]
        
        fields = ', '.join([f'LIST({field} ORDER BY {self.col_event_time})' for field in event_fields])

        field_names = [self.col_id] + event_fields
        
        query = f"""
            SELECT {self.col_id}, {fields}
            FROM {self.data_read_func}
            GROUP BY {self.col_id}
            """
    
        rel = duckdb.sql(query)

        while(True):
            recs = rel.fetchmany(1000)
            if not recs:
                break
            for rec in recs:
                feature_dict = dict(zip(field_names, rec))
                for fld in event_fields:
                    feature_dict[fld] = torch.tensor(feature_dict[fld])

                yield feature_dict

    def get_category_sizes(self, fields):
        field_cnt = ', '.join([f'COUNT(DISTINCT {field})' for field in fields])

        sizes = duckdb.sql(f"""SELECT {field_cnt} from {self.data_read_func}""").fetchone()

        return dict(zip(fields, sizes))
