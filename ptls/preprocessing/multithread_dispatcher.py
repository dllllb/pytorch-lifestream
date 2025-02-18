from typing import Union, List, Dict, Callable

import dask
import pandas as pd
from joblib import wrap_non_picklable_objects, parallel_backend
from pymonad.maybe import Maybe
from ptls.preprocessing.dask.dask_client import DaskServer

from ptls.preprocessing.util import determine_n_jobs


class DaskDispatcher:
    def __init__(self, n_jobs: int = -1):
        self.dask_client = DaskServer().client
        print(f"Link Dask Server - {self.dask_client.dashboard_link}")
        self.transformation_func = "fit_transform"
        self.n_jobs = determine_n_jobs(-1)

    def shutdown(self):
        # self.dask_client.close()
        # del self.dask_client
        del self

    def _multithread_eval(
            self,
            individuals_to_evaluate: Union[Dict, pd.DataFrame],
            objective_func: Union[Callable, Dict],
    ):
        if isinstance(objective_func, dict):
            evaluation_results = list(map(lambda func_name, func_impl:
                                          self.evaluate_single(
                                              self,
                                              data=pd.DataFrame(individuals_to_evaluate[func_name]),
                                              eval_func=func_impl
                                          ),
                                          objective_func.keys(), objective_func.values() 
                                          )
                                      )
            evaluation_results = dask.compute(*evaluation_results)
        else:
            evaluation_results = self.evaluate_single(self, data=individuals_to_evaluate, eval_func=objective_func)
        return evaluation_results

    def __multithread_eval(
            self,
            individuals_to_evaluate: Union[Dict, pd.DataFrame],
            objective_func: Union[Callable, Dict],
    ):
        """
        Evaluate individuals in parallel without using dask
        """
        #     without dask

        if isinstance(objective_func, dict):
            evaluation_results = [
                self.evaluate_single(
                    self,
                    # data=individuals_to_evaluate[func_name],
                    data=pd.DataFrame(individuals_to_evaluate[func_name]),
                    eval_func=func_impl,
                )
                for func_name, func_impl in objective_func.items()
            ]
        else:
            evaluation_results = self.evaluate_single(
                self, data=individuals_to_evaluate, eval_func=objective_func
            )
        return evaluation_results

    def evaluate(
            self,
            individuals: Union[List, Dict],
            objective_func: Union[Callable, List[Callable]],
    ):
        individuals_evaluated = Maybe.insert(individuals).maybe(
            default_value=None,
            extraction_function=lambda generation: self._multithread_eval(
                generation, objective_func
            ),
        )
        return individuals_evaluated

    @wrap_non_picklable_objects
    def evaluate_single(
            self, data: Union[pd.DataFrame, pd.Series], eval_func: Callable
    ):
        if self.transformation_func == "fit_transform":
            eval_res = eval_func.fit_transform(data)
        else:
            eval_res = eval_func.transform(data)
        return eval_res
