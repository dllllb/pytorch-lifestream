import pathlib
import timeit
from datetime import datetime
from typing import Optional, Tuple
from joblib import wrap_non_picklable_objects, parallel_backend
from pymonad.either import Either
from pymonad.maybe import Maybe

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

from ptls.preprocessing.dask.dask_client import DaskServer


class DaskDispatcher:
    def __init__(self):
        self.dask_client = DaskServer().client
        self.logger.info(
            f'LinK Dask Server - {self.dask_client.dashboard_link}')

    def shutdown(self):
        self.dask_client.close()
        del self.dask_client

    def _multithread_eval(self, individuals_to_evaluate):
        with parallel_backend(backend='dask',
                              n_jobs=self.n_jobs,
                              scatter=[individuals_to_evaluate]
                              ):
            evaluation_results = [self.evaluate_single(
                    graph=ind.graph,
                    uid_of_individual=ind.uid) for ind in individuals_to_evaluate]
        return evaluation_results

    def _eval_at_least_one(self, individuals):
        successful_evals = None
        for single_ind in individuals:
            try:
                evaluation_result = self.industrial_evaluate_single(
                    self, graph=single_ind.graph, uid_of_individual=single_ind.uid, with_time_limit=False)
                successful_evals = self.apply_evaluation_results(
                    [single_ind], [evaluation_result])
                if successful_evals:
                    break
            except Exception:
                successful_evals = None
        return successful_evals

    def evaluate(self, individuals):
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(
            individuals)

        # Evaluate individuals without valid fitness in parallel.
        self.n_jobs = determine_n_jobs(self._n_jobs, self.logger)

        individuals_evaluated = Maybe(individuals,
                                      monoid=[individuals, True]).then(
            lambda generation: self._multithread_eval(generation)). \
            then(lambda eval_res: self.apply_evaluation_results(individuals_to_evaluate, eval_res)).value
        successful_evals = Either(successful_evals,
                                  monoid=[individuals_evaluated, not successful_evals]).either(
            left_function=lambda x: x,
            right_function=lambda y: self._eval_at_least_one(y))
        return successful_evals

    # @delayed
    @wrap_non_picklable_objects
    def evaluate_single(self,
                        graph: OptGraph,
                        uid_of_individual: str,
                        with_time_limit: bool = True,
                        cache_key: Optional[str] = None,
                        logs_initializer: Optional[Tuple[int,
                                                         pathlib.Path]] = None) -> GraphEvalResult:
        if self._n_jobs != 1:
            IndustrialModels().setup_repository()

        graph = self.evaluation_cache.get(cache_key, graph)

        if with_time_limit and self.timer.is_time_limit_reached():
            return None
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)

        adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
        start_time = timeit.default_timer()
        fitness, graph = adapted_evaluate(graph)
        end_time = timeit.default_timer()
        eval_time_iso = datetime.now().isoformat()
        return eval_res
