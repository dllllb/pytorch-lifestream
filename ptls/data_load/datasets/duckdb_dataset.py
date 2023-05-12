from typing import List

import duckdb
import torch

class DuckDbDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            data_read_func: str,
            col_id: str,
            col_event_time: str,
            col_event_fields: List[str]):
        self.data_read_func = data_read_func
        self.col_id = col_id
        self.col_event_time = col_event_time
        self.col_event_fields = col_event_fields

    def __iter__(self):
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
