from __future__ import annotations

import json
import uuid
from typing import List

import math
from mlflow.entities import (
    Experiment,
    RunTag,
    Metric,
    Param,
    Run,
    RunStatus,
    LifecycleStage,
    ViewType,
    SourceType,
)
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities import PagedList
from mlflow.store.model_registry import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.tracking import (
    SEARCH_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.file_utils import local_file_uri_to_path, mkdir
from mlflow.utils.mlflow_tags import (
    _get_run_name_from_tags,
    MLFLOW_RUN_NAME,
    MLFLOW_LOGGED_MODELS,
)
from mlflow.utils.name_utils import _generate_random_name
from mongoengine import connect, BulkWriteError
from mongoengine.queryset.visitor import Q
from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path, resolve_uri_if_local, is_local_uri
from mlflow.utils.search_utils import SearchUtils, SearchExperimentsUtils
from mlflow.utils.validation import (
    _validate_batch_log_limits,
    _validate_batch_log_data,
    _validate_run_id,
    _validate_experiment_name,
    _validate_experiment_tag,
    _validate_metric,
    _validate_param_keys_unique,
    _validate_tag,
)
from mlflow.utils.time_utils import get_current_time_millis

from mlflow_mongostore.models import (
    MongoExperiment,
    MongoRun,
    MongoMetric,
    MongoParam,
    MongoTag,
    MongoExperimentTag,
    SequenceId,
    MongoLatestMetric,
)

RunStatusTypes = [
    RunStatus.to_string(RunStatus.SCHEDULED),
    RunStatus.to_string(RunStatus.FAILED),
    RunStatus.to_string(RunStatus.FINISHED),
    RunStatus.to_string(RunStatus.RUNNING),
    RunStatus.to_string(RunStatus.KILLED),
]


def _like_to_regex(like_pattern: str):
    like_pattern = "^" + like_pattern + "$"
    return like_pattern.replace("%", ".*")


def _get_filter_query(attr, comp, value):
    if comp == ">":
        return Q(**{f"{attr}__gt": value})
    elif comp == ">=":
        return Q(**{f"{attr}__gte": value})
    elif comp == "!=":
        return Q(**{f"{attr}__ne": value})
    elif comp == "=":
        return Q(**{f"{attr}": value})
    elif comp == "<":
        return Q(**{f"{attr}__lt": value})
    elif comp == "<=":
        return Q(**{f"{attr}__lte": value})
    elif comp == "LIKE":
        return Q(**{f"{attr}__regex": _like_to_regex(value)})
    elif comp == "ILIKE":
        return Q(**{f"{attr}__iregex": _like_to_regex(value)})
    elif comp == "IN":
        return Q(**{f"{attr}__in": value})
    elif comp == "NOT IN":
        return Q(**{f"{attr}__nin": value})


def _get_list_contains_query(key, val, comp, list_field_name):
    value_filter = {}

    if comp == ">":
        value_filter = {"$gt": val}
    elif comp == ">=":
        value_filter = {"$gte": val}
    elif comp == "!=":
        value_filter = {"$ne": val}
    elif comp == "=":
        value_filter = val
    elif comp == "<":
        value_filter = {"$lt": val}
    elif comp == "<=":
        value_filter = {"$lte": val}
    elif comp == "LIKE":
        value_filter = {"$regex": _like_to_regex(val)}
    elif comp == "ILIKE":
        value_filter = {"$regex": _like_to_regex(val), "$options": "i"}

    return Q(**{f"{list_field_name}__match": {"key": key, "value": value_filter}})


def _get_metrics_contains_query(key, val, comp):
    value_filter = {}

    if comp == ">":
        value_filter = "__gt"
    elif comp == ">=":
        value_filter = "__gte"
    elif comp == "!=":
        value_filter = "__ne"
    elif comp == "=":
        value_filter = ""
    elif comp == "<":
        value_filter = "__lt"
    elif comp == "<=":
        value_filter = "__lte"
    return Q(**{"latest_metrics__match": {"key": key, f"value{value_filter}": val}})


def _order_by_clause(key, ascending):
    if key == "experiment_id":
        key = "exp_id"
    if ascending:
        return f"+{key}"
    return f"-{key}"


def _get_search_experiments_filter_clauses(parsed_filters):
    _filter = Q()
    for f in parsed_filters:
        type_ = f["type"]
        key = f["key"]
        comparator = f["comparator"]
        value = f["value"]
        if type_ == "attribute":
            if SearchExperimentsUtils.is_string_attribute(
                type_, key, comparator
            ) and comparator not in ("=", "!=", "LIKE", "ILIKE"):
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for string attribute: {comparator}"
                )
            if SearchExperimentsUtils.is_numeric_attribute(
                type_, key, comparator
            ) and comparator not in ("=", "!=", "<", "<=", ">", ">="):
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for numeric attribute: {comparator}"
                )
            _filter &= _get_filter_query(key, comparator, value)
        elif type_ == "tag":
            if comparator not in ("=", "!=", "LIKE", "ILIKE"):
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for tag: {comparator}"
                )
            _filter &= _get_list_contains_query(
                key=key, val=value, comp=comparator, list_field_name="tags"
            )
        else:
            raise MlflowException.invalid_parameter_value(
                f"Invalid token type: {type_}"
            )

    return _filter


def _get_search_experiments_order_by_clauses(order_by):
    order_by_clauses = []
    for type_, key, ascending in map(
        SearchExperimentsUtils.parse_order_by_for_search_experiments,
        order_by or ["creation_time DESC", "experiment_id ASC"],
    ):
        if type_ == "attribute":
            order_by_clauses.append((key, ascending))
        else:
            raise MlflowException.invalid_parameter_value(
                f"Invalid order_by entity: {type_}"
            )

    # Add a tie-breaker
    if not any(col == "experiment_id" for col, _ in order_by_clauses):
        order_by_clauses.append(("experiment_id", False))

    return [_order_by_clause(col, ascending) for col, ascending in order_by_clauses]


def _get_search_run_filter_clauses(parsed_filters):
    _filter = Q()

    for f in parsed_filters:
        type_ = f.get("type")
        key = f.get("key")
        value = f.get("value")
        comparator = f.get("comparator").upper()

        key = SearchUtils.translate_key_alias(key)

        if SearchUtils.is_string_attribute(
            type_, key, comparator
        ) or SearchUtils.is_numeric_attribute(type_, key, comparator):
            if key == "run_name":
                # Treat "attributes.run_name == <value>" as "tags.`mlflow.runName` == <value>".
                # The name column in the runs table is empty for runs logged in MLFlow <= 1.29.0.
                _filter &= _get_list_contains_query(
                    key=MLFLOW_RUN_NAME,
                    val=value,
                    comp=comparator,
                    list_field_name="tags",
                )
            else:
                key = MongoRun.get_attribute_name(key)
                _filter &= _get_filter_query(key, comparator, value)
        else:
            if SearchUtils.is_metric(type_, comparator):
                value = float(value)
                _filter &= _get_metrics_contains_query(
                    key=key, val=value, comp=comparator
                )
            elif SearchUtils.is_param(type_, comparator):
                entity = "params"
                _filter &= _get_list_contains_query(
                    key=key, val=value, comp=comparator, list_field_name=entity
                )
            elif SearchUtils.is_tag(type_, comparator):
                entity = "tags"
                _filter &= _get_list_contains_query(
                    key=key, val=value, comp=comparator, list_field_name=entity
                )
            else:
                raise MlflowException(
                    "Invalid search expression type '%s'" % type_,
                    error_code=INVALID_PARAMETER_VALUE,
                )

    return _filter


def _get_next_exp_id(start_over=False):
    if start_over:
        seq = SequenceId(collection_name="mlflow-experiments", sequence_value=0)
        seq.save()
        return "0"

    return str(
        SequenceId._get_collection().find_one_and_update(
            filter={"_id": "mlflow-experiments"},
            update={"$inc": {"sequence_value": 1}},
            new=True,
        )["sequence_value"]
    )


class MongoStore(AbstractStore):
    ARTIFACTS_FOLDER_NAME = "artifacts"

    DEFAULT_EXPERIMENT_ID = "0"

    filter_key = {
        ">": ["range", "must"],
        ">=": ["range", "must"],
        "=": ["term", "must"],
        "!=": ["term", "must_not"],
        "<=": ["range", "must"],
        "<": ["range", "must"],
        "LIKE": ["wildcard", "must"],
        "ILIKE": ["wildcard", "must"],
    }

    def __init__(self, store_uri: str, artifact_uri) -> None:
        super(MongoStore, self).__init__()

        self.is_plugin = True

        if artifact_uri is None:
            artifact_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
        self.artifact_root_uri = resolve_uri_if_local(artifact_uri)

        parsed_uri = urllib.parse.urlparse(store_uri)
        self.__db_name = parsed_uri.path.replace("/", "")
        self.__conn = connect(
            db=self.__db_name, host=f"{parsed_uri.scheme}://{parsed_uri.netloc}"
        )

        if is_local_uri(artifact_uri):
            mkdir(local_file_uri_to_path(artifact_uri))

        if len(self.search_experiments(view_type=ViewType.ALL)) == 0:
            self._create_default_experiment()

    # ##################################################################################################################
    # ############################################# EXPERIMENT APIs ####################################################
    # ##################################################################################################################

    def get_experiment(self, experiment_id: str) -> Experiment:
        return self._get_experiment(experiment_id).to_mlflow_entity()

    def get_experiment_by_name(self, experiment_name):
        exp = self._get_experiment_by_name(experiment_name)
        return exp.to_mlflow_entity() if exp is not None else None

    def create_experiment(self, name, artifact_location=None, tags=None):
        _validate_experiment_name(name)

        existing_names = self._list_experiments_name()
        if name in existing_names:
            raise MlflowException(
                f"Experiment(name={name}) already exists", INVALID_PARAMETER_VALUE
            )

        tags_dict = {}
        if tags is not None:
            for tag in tags:
                tags_dict[tag.key] = tag.value
        exp_tags = [
            MongoExperimentTag(key=key, value=value) for key, value in tags_dict.items()
        ]

        curr_time = get_current_time_millis()
        experiment = MongoExperiment(
            exp_id=_get_next_exp_id(),
            name=name,
            lifecycle_stage=LifecycleStage.ACTIVE,
            artifact_location=artifact_location,
            tags=exp_tags,
            creation_time=curr_time,
            last_update_time=curr_time,
        )
        experiment.save()
        if not artifact_location:
            artifact_location = self._get_artifact_location(experiment.id)
        experiment.update(artifact_location=artifact_location)
        return str(experiment.id)

    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        filter_string=None,
        order_by=None,
        page_token=None,
    ):
        experiments, next_page_token = self._search_experiments(
            view_type, max_results, filter_string, order_by, page_token
        )
        return PagedList(experiments, next_page_token)

    def delete_experiment(self, experiment_id):
        experiment = self._get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "Cannot delete an already deleted experiment.", INVALID_STATE
            )
        for run in self._list_run_by_exp_id(experiment_id):
            run.update(
                lifecycle_stage=LifecycleStage.DELETED,
                deleted_time=get_current_time_millis(),
            )
        experiment.update(
            lifecycle_stage=LifecycleStage.DELETED,
            last_update_time=get_current_time_millis(),
        )

    def restore_experiment(self, experiment_id):
        experiment = self._get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.DELETED:
            raise MlflowException(
                "Cannot restore an already active experiment.", INVALID_STATE
            )

        for run in self._list_run_by_exp_id(experiment_id):
            run.update(lifecycle_stage=LifecycleStage.ACTIVE, deleted_time=None)
        experiment.update(
            lifecycle_stage=LifecycleStage.ACTIVE,
            last_update_time=get_current_time_millis(),
        )

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        experiment = self._get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "Cannot rename a non-active experiment.", INVALID_STATE
            )
        experiment.update(name=new_name, last_update_time=get_current_time_millis())

    def record_logged_model(self, run_id, mlflow_model):
        from mlflow.models import Model

        if not isinstance(mlflow_model, Model):
            raise TypeError(
                "Argument 'mlflow_model' should be mlflow.models.Model, got '{}'".format(
                    type(mlflow_model)
                )
            )
        model_dict = mlflow_model.to_dict()
        run = self._get_run(run_uuid=run_id)
        self._check_run_is_active(run)
        previous_tag = [t for t in run.tags if t.key == MLFLOW_LOGGED_MODELS]
        if previous_tag:
            value = json.dumps(json.loads(previous_tag[0].value) + [model_dict])
        else:
            value = json.dumps([model_dict])
        _validate_tag(MLFLOW_LOGGED_MODELS, value)
        self._set_tag(run, RunTag(key=MLFLOW_LOGGED_MODELS, value=value))

    def set_experiment_tag(self, exp_id, tag):
        _validate_experiment_tag(tag.key, tag.value)
        experiment = self._get_experiment(exp_id)
        self._check_experiment_is_active(experiment)
        experiment.update(push__tags=MongoExperimentTag(key=tag.key, value=tag.value))

    def _search_experiments(
        self, view_type, max_results, filter_string, order_by, page_token
    ):
        def compute_next_token(current_size):
            next_token = None
            if max_results + 1 == current_size:
                final_offset = offset + max_results
                next_token = SearchExperimentsUtils.create_page_token(final_offset)

            return next_token

        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                f"Invalid value for max_results. It must be at most {SEARCH_MAX_RESULTS_THRESHOLD},"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        lifecycle_stages = set(LifecycleStage.view_type_to_stages(view_type))
        _filter = Q(**{"lifecycle_stage__in": lifecycle_stages})

        parsed_filters = SearchExperimentsUtils.parse_search_filter(filter_string)
        _filter &= _get_search_experiments_filter_clauses(parsed_filters)

        order_by_clauses = _get_search_experiments_order_by_clauses(order_by)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)

        experiments = MongoExperiment.objects(_filter).order_by(*order_by_clauses)[
            offset : max_results + offset + 1
        ]
        experiments = [e.to_mlflow_entity() for e in experiments]

        next_page_token = compute_next_token(len(experiments))

        return experiments[:max_results], next_page_token

    def _list_run_by_exp_id(self, experiment_id):
        return MongoRun.objects(experiment_id=experiment_id)

    def _list_experiments_name(self) -> List[str]:
        return [obj.name for obj in MongoExperiment.objects.only("name")]

    def _get_experiment(self, experiment_id: str) -> MongoExperiment:
        try:
            experiment = MongoExperiment.objects(exp_id=experiment_id)[0]
        except IndexError:
            raise MlflowException(
                "No Experiment with id={} exists".format(experiment_id),
                RESOURCE_DOES_NOT_EXIST,
            )
        return experiment

    def _get_experiment_by_name(self, experiment_name: str) -> [MongoExperiment | None]:
        try:
            experiment = MongoExperiment.objects(name=experiment_name)[0]
        except IndexError:
            return None
        return experiment

    def _check_experiment_is_active(self, experiment: MongoExperiment) -> None:
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "The experiment {} must be in the 'active' state. "
                "Current state is {}.".format(
                    experiment.id, experiment.lifecycle_stage
                ),
                INVALID_PARAMETER_VALUE,
            )

    # ##################################################################################################################
    # ################################################# RUN APIs #######################################################
    # ##################################################################################################################
    def create_run(self, experiment_id, user_id, start_time, tags, run_name) -> Run:
        experiment = self._get_experiment(experiment_id)
        self._check_experiment_is_active(experiment)

        run_id = uuid.uuid4().hex
        artifact_location = append_to_uri_path(
            experiment.artifact_location, run_id, MongoStore.ARTIFACTS_FOLDER_NAME
        )

        tags = tags or []
        run_name_tag = _get_run_name_from_tags(tags)
        if run_name and run_name_tag and (run_name != run_name_tag):
            raise MlflowException(
                "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                f"different values (run_name='{run_name}', mlflow.runName='{run_name_tag}').",
                INVALID_PARAMETER_VALUE,
            )

        run_name = run_name or run_name_tag or _generate_random_name()
        if not run_name_tag:
            tags.append(RunTag(key=MLFLOW_RUN_NAME, value=run_name))

        run_tags = [MongoTag(key=tag.key, value=tag.value) for tag in tags]
        run = MongoRun(
            name=run_name,
            artifact_uri=artifact_location,
            run_uuid=run_id,
            experiment_id=experiment_id,
            source_type=SourceType.to_string(SourceType.UNKNOWN),
            source_name="",
            entry_point_name="",
            user_id=user_id,
            status=RunStatus.to_string(RunStatus.RUNNING),
            start_time=start_time,
            end_time=None,
            deleted_time=None,
            source_version="",
            lifecycle_stage=LifecycleStage.ACTIVE,
            tags=run_tags,
        )

        run.save()
        return run.to_mlflow_entity()

    def get_run(self, run_id):
        return self._get_run(run_id).to_mlflow_entity()

    def update_run_info(self, run_id, run_status, end_time, run_name):
        run = self._get_run(run_id)
        self._check_run_is_active(run)
        run.update(status=RunStatus.to_string(run_status), end_time=end_time)
        if run_name:
            run.update(name=run_name)
            num_updates = run.tags.filter(key=MLFLOW_RUN_NAME).update(
                key=MLFLOW_RUN_NAME, value=run_name
            )
            if num_updates == 0:
                run.tags.append(MongoTag(key=MLFLOW_RUN_NAME, value=run_name))
            run.save()

        run.reload()
        return run.to_mlflow_entity().info

    def delete_run(self, run_id):
        run = self._get_run(run_id)
        run.update(
            lifecycle_stage=LifecycleStage.DELETED,
            deleted_time=get_current_time_millis(),
        )

    def restore_run(self, run_id):
        run = self._get_run(run_id)
        run.update(lifecycle_stage=LifecycleStage.ACTIVE, deleted_time=None)

    def log_batch(
        self,
        run_id: str,
        metrics: List[Metric],
        params: List[Param],
        tags: List[RunTag],
    ) -> None:
        _validate_run_id(run_id)
        _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        _validate_param_keys_unique(params)

        run = self._get_run(run_uuid=run_id)
        self._check_run_is_active(run)
        try:
            if metrics:
                self._log_metrics(run, metrics)
            for param in params:
                self._log_param(run, param)
            for tag in tags:
                self._set_tag(run, tag)
            run.save()
        except MlflowException as e:
            raise e
        except Exception as e:
            raise MlflowException(e)

    def get_metric_history(self, run_id, metric_key, max_results=None, page_token=None):
        if page_token is not None:
            raise MlflowException(
                "The SQLAlchemyStore backend does not support pagination for the "
                f"`get_metric_history` API. Supplied argument `page_token` '{page_token}' must be "
                "`None`."
            )

        metrics = MongoMetric.objects(run_uuid=run_id, key=metric_key)
        return PagedList([metric.to_mlflow_entity() for metric in metrics], None)

    def delete_tag(self, run_id, key):
        run = self._get_run(run_id)
        self._check_run_is_active(run)
        tags = run.get_tags_by_key(key)
        if len(tags) == 0:
            raise MlflowException(
                f"No tag with name: {key} in run with id {run.id}",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        elif len(tags) > 1:
            raise MlflowException(
                "Bad data in database - tags for a specific run must have "
                "a single unique value. "
                "See https://mlflow.org/docs/latest/tracking.html#adding-tags-to-runs",
                error_code=INVALID_STATE,
            )
        run.update(pull__tags=tags[0])

    def _log_metrics(self, run, metrics):
        metric_instances = []
        seen = set()
        for metric in metrics:
            metric, value, is_nan = self._get_metric_value_details(metric)
            if metric not in seen:
                metric_instances.append(
                    MongoMetric(
                        run_uuid=run.id,
                        key=metric.key,
                        value=value,
                        timestamp=metric.timestamp,
                        step=metric.step,
                        is_nan=is_nan,
                    )
                )
            seen.add(metric)

        def _insert_metrics(metric_instances):
            MongoMetric.objects.insert(metric_instances, load_bulk=False)
            for m in metric_instances:
                self._update_latest_metric_if_necessary(run, m)

        try:
            _insert_metrics(metric_instances)
        except BulkWriteError:
            pass

    def _get_metric_value_details(self, metric):
        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)
        is_nan = math.isnan(metric.value)
        if is_nan:
            value = 0
        elif math.isinf(metric.value):
            #  NB: Sql can not represent Infs = > We replace +/- Inf with max/min 64b float value
            value = (
                1.7976931348623157e308 if metric.value > 0 else -1.7976931348623157e308
            )
        else:
            value = metric.value
        return metric, value, is_nan

    def _update_latest_metric_if_necessary(self, run, logged_metric):
        def _compare_metrics(metric_a, metric_b):
            """
            :return: True if ``metric_a`` is strictly more recent than ``metric_b``, as determined
                     by ``step``, ``timestamp``, and ``value``. False otherwise.
            """
            return (metric_a.step, metric_a.timestamp, metric_a.value) > (
                metric_b.step,
                metric_b.timestamp,
                metric_b.value,
            )

        new_latest_metric = MongoLatestMetric(
            key=logged_metric.key,
            value=logged_metric.value,
            timestamp=logged_metric.timestamp,
            step=logged_metric.step,
            is_nan=logged_metric.is_nan,
        )

        latest_metric_exist = False
        for i, latest_metric in enumerate(run.latest_metrics):
            if latest_metric.key == logged_metric.key:
                latest_metric_exist = True
                if _compare_metrics(new_latest_metric, latest_metric):
                    run.latest_metrics[i] = new_latest_metric
        if not latest_metric_exist:
            run.update(push__latest_metrics=new_latest_metric)

    def _log_param(self, run, param):
        existing_param = run.get_param_by_key(param.key)
        if existing_param:
            if existing_param.value != param.value:
                raise MlflowException(
                    "Changing param values is not allowed. Params were already"
                    f" logged='{param}' for run ID='{run.id}'.",
                    INVALID_PARAMETER_VALUE,
                )
            else:
                return
        new_param = MongoParam(key=param.key, value=param.value)
        run.update(push__params=new_param)
        run.reload()

    def _set_tag(self, run, tag):
        if tag.key == MLFLOW_RUN_NAME:
            run.update(name=tag.value)

        existing = run.tags.filter(key=tag.key)
        if existing.count() == 0:
            new_tag = MongoTag(key=tag.key, value=tag.value)
            run.update(push__tags=new_tag)
        else:
            existing.update(key=tag.key, value=tag.value)
            run.save()

    def _search_runs(
        self,
        experiment_ids,
        filter_string,
        run_view_type,
        max_results,
        order_by,
        page_token,
    ):
        # def compute_next_token(current_size):
        #     next_token = None
        #     if max_results == current_size:
        #         final_offset = offset + max_results
        #         next_token = SearchUtils.create_page_token(final_offset)
        #
        #     return next_token
        experiment_ids = list(experiment_ids)
        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at "
                f"most {SEARCH_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        lifecycle_stages = list(set(LifecycleStage.view_type_to_stages(run_view_type)))
        _filter = Q(experiment_id__in=experiment_ids)
        _filter &= Q(lifecycle_stage__in=lifecycle_stages)

        parsed_filters = SearchUtils.parse_search_filter(filter_string)
        _filter &= _get_search_run_filter_clauses(parsed_filters)
        # order_by_clauses = _get_search_run_order_by_clauses(order_by)
        # offset = SearchUtils.parse_start_offset_from_page_token(page_token)

        runs = MongoRun.objects(_filter)
        runs = [r.to_mlflow_entity() for r in runs]
        runs = SearchUtils.sort(runs, order_by)
        runs, next_page_token = SearchUtils.paginate(runs, page_token, max_results)

        return runs, next_page_token
        # next_page_token = compute_next_token(len(runs))
        #
        # return runs[:max_results], next_page_token

    def _is_valid_run(self, run: MongoRun, stages, parsed_filters):
        if run.lifecycle_stage not in stages:
            return False

        for f in parsed_filters:
            key_type = f.get("type")
            key_name = f.get("key")
            value = f.get("value")
            comparator = f.get("comparator").upper()

            if SearchUtils.is_string_attribute(
                key_type, key_name, comparator
            ) or SearchUtils.is_numeric_attribute(key_type, key_name, comparator):
                return run.match_attr(key_name, comparator, value)
            else:
                if SearchUtils.is_metric(key_type, comparator):
                    value = float(value)
                    return run.match_metric(key_name, comparator, value)
                elif SearchUtils.is_param(key_type, comparator):
                    return run.match_param(key_name, comparator, value)
                elif SearchUtils.is_tag(key_type, comparator):
                    return run.match_tag(key_name, comparator, value)
                else:
                    raise MlflowException(
                        "Invalid search expression type '%s'" % key_type,
                        error_code=INVALID_PARAMETER_VALUE,
                    )

    def _get_run(self, run_uuid: str) -> MongoRun:
        runs = MongoRun.objects(run_uuid=run_uuid)

        if len(runs) == 0:
            raise MlflowException(
                "Run with id={} not found".format(run_uuid), RESOURCE_DOES_NOT_EXIST
            )
        if len(runs) > 1:
            raise MlflowException(
                "Expected only 1 run with id={}. Found {}.".format(run_uuid, len(runs)),
                INVALID_STATE,
            )
        return runs[0]

    def _hard_delete_run(self, run_id):
        MongoRun.objects(run_uuid=run_id).delete()

    def _get_deleted_runs(self, older_than=0):
        current_time = get_current_time_millis()
        return [
            r.run_uuid
            for r in MongoRun.objects(
                lifecycle_stage=LifecycleStage.DELETED,
                deleted_time__lte=(current_time - older_than),
            )
        ]

    def _check_run_is_active(self, run: MongoRun) -> None:
        if run.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "The run {} must be in the 'active' state. Current state is {}.".format(
                    run.id, run.lifecycle_stage
                ),
                INVALID_PARAMETER_VALUE,
            )

    def _create_default_experiment(self):
        """
        MLflow UI and client code expects a default experiment with ID 0.
        This method uses SQL insert statement to create the default experiment as a hack, since
        experiment table uses 'experiment_id' column is a PK and is also set to auto increment.
        MySQL and other implementation do not allow value '0' for such cases.

        ToDo: Identify a less hacky mechanism to create default experiment 0
        """
        SequenceId.drop_collection()
        creation_time = get_current_time_millis()
        exp = MongoExperiment(
            exp_id=_get_next_exp_id(start_over=True),
            name=Experiment.DEFAULT_EXPERIMENT_NAME,
            artifact_location=self._get_artifact_location(0),
            lifecycle_stage=LifecycleStage.ACTIVE,
            creation_time=creation_time,
            last_update_time=creation_time,
        )
        exp.save()

    def _get_artifact_location(self, experiment_id):
        return append_to_uri_path(self.artifact_root_uri, str(experiment_id))
