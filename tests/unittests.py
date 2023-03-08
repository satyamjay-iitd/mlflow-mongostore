from unittest.mock import MagicMock

import pytest
import mock
from types import SimpleNamespace

from mlflow.entities import (RunTag, Metric, Param, RunStatus,
                             LifecycleStage, ViewType, ExperimentTag)

from mlflow_mongostore.mongo_store import MongoStore
from mlflow_mongostore.models import (MongoExperiment, MongoRun, MongoMetric,
                                      MongoMetric, MongoParam,
                                      MongoTag, MongoExperimentTag)

experiment = MongoExperiment(name="name",
                             lifecycle_stage=LifecycleStage.ACTIVE,
                             artifact_location="artifact_location")
deleted_experiment = MongoExperiment(name="name",
                                     lifecycle_stage=LifecycleStage.DELETED,
                                     artifact_location="artifact_location")

experiment_tag = ExperimentTag(key="tag1", value="val1")
elastic_experiment_tag = MongoExperimentTag(key="tag1", value="val1")

run = MongoRun(
               run_id="1",
               experiment_id="experiment_id", user_id="user_id",
               status=RunStatus.to_string(RunStatus.RUNNING),
               start_time=1, end_time=None,
               lifecycle_stage=LifecycleStage.ACTIVE, artifact_uri="artifact_location",
               latest_metrics=[MongoMetric(
                   key="metric1", value=1, timestamp=1, step=1, is_nan=False)],
               params=[MongoParam(key="param1", value="val1")],
               tags=[MongoTag(key="tag1", value="val1")])

deleted_run = MongoRun(
                       run_id="1",
                       experiment_id="experiment_id", user_id="user_id",
                       status=RunStatus.to_string(RunStatus.RUNNING),
                       start_time=1, end_time=None,
                       lifecycle_stage=LifecycleStage.DELETED,
                       artifact_uri="artifact_location",
                       latest_metrics=[MongoMetric(
                           key="metric1", value=1, timestamp=1, step=1, is_nan=False)],
                       params=[MongoParam(key="param1", value="val1")],
                       tags=[MongoTag(key="tag1", value="val1")])

elastic_metric = MongoMetric(key="metric2", value=2, timestamp=1,
                             step=1, is_nan=False, run_id="1")

metric = Metric(key="metric2", value=2, timestamp=1, step=1)

elastic_param = MongoParam(key="param2", value="val2")
param = Param(key="param2", value="val2")

elastic_tag = MongoTag(key="tag2", value="val2")
tag = RunTag(key="tag2", value="val2")


# @mock.patch('mongoengine.queryset.base.BaseQuerySet.only')
# @pytest.mark.usefixtures('create_store')
# def test_list_experiments(mongo_obj_filter_mock, create_store):
#     hit = {"name": "name",
#            "lifecycle_stage": LifecycleStage.ACTIVE,
#            "artifact_location": "artifact_location"}
#     response = [SimpleNamespace(**hit)]
#     mongo_obj_filter_mock.return_value.execute = mock.MagicMock(return_value=response)
#     real_experiments = create_store._list_experiments()
#     mongo_obj_filter_mock.assert_called_once_with(
#         "terms", lifecycle_stage=LifecycleStage.view_type_to_stages(ViewType.ACTIVE_ONLY))
#     mongo_obj_filter_mock.return_value.execute.assert_called_once_with()
#     mock_experiments = [create_store._hit_to_mlflow_experiment(e) for e in response]
#
#     assert real_experiments[0].__dict__ == mock_experiments[0].__dict__

import mongoengine


@mock.patch('mlflow_mongostore.models.MongoExperiment.objects')
@pytest.mark.usefixtures('mongo_store')
def test_get_experiment(mongo_exp_objs, mongo_store):
    mongo_exp_objs.return_value = [experiment]
    real_experiment = mongo_store.get_experiment("1")
    MongoExperiment.objects.assert_called_once_with(name="1")
    assert experiment.to_mlflow_entity().__dict__ == real_experiment.__dict__


@mock.patch('mlflow_mongostore.models.MongoExperiment.objects')
@pytest.mark.usefixtures('mongo_store')
def test_delete_experiment(mongo_exp_objs, mongo_store):
    mongo_exp_objs.return_value = [experiment]
    experiment.update = mock.MagicMock()
    mongo_store.delete_experiment("1")
    mongo_exp_objs.assert_called_once_with(name="1")
    experiment.update.assert_called_once_with(lifecycle_stage=LifecycleStage.DELETED)


@mock.patch('mlflow_elasticsearchstore.models.MongoExperiment.get')
@pytest.mark.usefixtures('create_store')
def test_restore_experiment(elastic_experiment_get_mock, create_store):
    elastic_experiment_get_mock.return_value = deleted_experiment
    deleted_experiment.update = mock.MagicMock()
    create_store.restore_experiment("1")
    elastic_experiment_get_mock.assert_called_once_with(id="1")
    deleted_experiment.update.assert_called_once_with(
        refresh=True, lifecycle_stage=LifecycleStage.ACTIVE)


@mock.patch('mlflow_elasticsearchstore.models.MongoExperiment.get')
@pytest.mark.usefixtures('create_store')
def test_rename_experiment(elastic_experiment_get_mock, create_store):
    elastic_experiment_get_mock.return_value = experiment
    experiment.update = mock.MagicMock()
    create_store.rename_experiment("1", "new_name")
    elastic_experiment_get_mock.assert_called_once_with(id="1")
    experiment.update.assert_called_once_with(refresh=True, name="new_name")


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.save')
@mock.patch('mlflow_elasticsearchstore.models.MongoExperiment.get')
@mock.patch('uuid.uuid4')
@pytest.mark.usefixtures('create_store')
def test_create_run(uuid_mock, elastic_experiment_get_mock,
                    elastic_run_save_mock, create_store):
    uuid_mock.return_value = SimpleNamespace(hex='run_id')
    elastic_experiment_get_mock.return_value = experiment
    real_run = create_store.create_run(experiment_id="1", user_id="user_id", start_time=1, tags=[])
    uuid_mock.assert_called_once_with()
    elastic_experiment_get_mock.assert_called_once_with(id="1")
    elastic_run_save_mock.assert_called_once_with()
    assert real_run._info.experiment_id == "1"
    assert real_run._info.user_id == "user_id"
    assert real_run._info.start_time == 1
    assert real_run._data.tags == {}
    assert real_run._info.run_id == "run_id"


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.get')
@pytest.mark.usefixtures('create_store')
def test_delete_run(elastic_run_get_mock, create_store):
    elastic_run_get_mock.return_value = run
    run.update = mock.MagicMock()
    create_store.delete_run("1")
    elastic_run_get_mock.assert_called_once_with(id="1")
    run.update.assert_called_once_with(lifecycle_stage=LifecycleStage.DELETED)


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.get')
@pytest.mark.usefixtures('create_store')
def test_restore_run(elastic_run_get_mock, create_store):
    elastic_run_get_mock.return_value = deleted_run
    deleted_run.update = mock.MagicMock()
    create_store.restore_run("1")
    elastic_run_get_mock.assert_called_once_with(id="1")
    deleted_run.update.assert_called_once_with(lifecycle_stage=LifecycleStage.ACTIVE)


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.get')
@pytest.mark.usefixtures('create_store')
def test_update_run_info(elastic_run_get_mock, create_store):
    elastic_run_get_mock.return_value = run
    run.update = mock.MagicMock()
    create_store.update_run_info("1", RunStatus.FINISHED, 2)
    elastic_run_get_mock.assert_called_once_with(id="1")
    run.update.assert_called_once_with(
        status=RunStatus.to_string(RunStatus.FINISHED), end_time=2)


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.get')
@pytest.mark.usefixtures('create_store')
def test__get_run(elastic_run_get_mock, create_store):
    elastic_run_get_mock.return_value = run
    real_run = create_store._get_run("1")
    MongoRun.get.assert_called_once_with(id="1")
    assert run == real_run


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.get')
@pytest.mark.usefixtures('create_store')
def test_get_run(elastic_run_get_mock, create_store):
    elastic_run_get_mock.return_value = run
    real_run = create_store.get_run("1")
    MongoRun.get.assert_called_once_with(id="1")
    assert run.to_mlflow_entity()._info == real_run._info
    assert run.to_mlflow_entity()._data._metrics == real_run._data._metrics
    assert run.to_mlflow_entity()._data._params == real_run._data._params
    assert run.to_mlflow_entity()._data._tags == real_run._data._tags


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.get')
@mock.patch('mlflow_elasticsearchstore.models.MongoMetric.save')
@mock.patch('mlflow_elasticsearchstore.elasticsearch_store.MongosearchStore.'
            '_update_latest_metric_if_necessary')
@pytest.mark.usefixtures('create_store')
def test_log_metric(_update_latest_metric_if_necessary_mock,
                    elastic_metric_save_mock, elastic_run_get_mock, create_store):
    elastic_run_get_mock.return_value = run
    run.update = mock.MagicMock()
    create_store.log_metric("1", metric)
    elastic_run_get_mock.assert_called_once_with(id="1")
    _update_latest_metric_if_necessary_mock.assert_called_once_with(elastic_metric, run)
    elastic_metric_save_mock.assert_called_once_with()
    run.update.assert_called_once_with(latest_metrics=run.latest_metrics)


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.get')
@pytest.mark.usefixtures('create_store')
def test_log_param(elastic_run_get_mock, create_store):
    elastic_run_get_mock.return_value = run
    run.params = mock.MagicMock()
    run.params.append = mock.MagicMock()
    run.update = mock.MagicMock()
    create_store.log_param("1", param)
    elastic_run_get_mock.assert_called_once_with(id="1")
    run.params.append.assert_called_once_with(elastic_param)
    run.update.assert_called_once_with(params=run.params)


@mock.patch('mlflow_elasticsearchstore.models.MongoExperiment.get')
@pytest.mark.usefixtures('create_store')
def test_set_experiment_tag(elastic_experiment_get_mock, create_store):
    elastic_experiment_get_mock.return_value = experiment
    experiment.tags = mock.MagicMock()
    experiment.tags.append = mock.MagicMock()
    experiment.update = mock.MagicMock()
    create_store.set_experiment_tag("1", experiment_tag)
    elastic_experiment_get_mock.assert_called_once_with(id="1")
    experiment.tags.append.assert_called_once_with(elastic_experiment_tag)
    experiment.update.assert_called_once_with(tags=experiment.tags)


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.get')
@pytest.mark.usefixtures('create_store')
def test_set_tag(elastic_run_get_mock, create_store):
    elastic_run_get_mock.return_value = run
    run.tags = mock.MagicMock()
    run.tags.append = mock.MagicMock()
    run.update = mock.MagicMock()
    create_store.set_tag("1", tag)
    elastic_run_get_mock.assert_called_once_with(id="1")
    run.tags.append.assert_called_once_with(elastic_tag)
    run.update.assert_called_once_with(tags=run.tags)


@pytest.mark.parametrize("test_elastic_metric,test_elastic_latest_metrics",
                         [(MongoMetric(key="metric1", value=2, timestamp=1, step=2, is_nan=False),
                           [MongoMetric(key="metric1", value=2, timestamp=1,
                                              step=2, is_nan=False)]),
                          (MongoMetric(key="metric2", value=2, timestamp=1, step=1, is_nan=False),
                           [MongoMetric(key="metric1", value=2, timestamp=1,
                                              step=2, is_nan=False),
                            MongoMetric(key="metric2", value=2, timestamp=1,
                                              step=1, is_nan=False)])])
@pytest.mark.usefixtures('create_store')
def test__update_latest_metric_if_necessary(test_elastic_metric, test_elastic_latest_metrics,
                                            create_store):
    create_store._update_latest_metric_if_necessary(test_elastic_metric, run)
    assert run.latest_metrics == test_elastic_latest_metrics


@pytest.mark.usefixtures('create_store')
def test__build_columns_to_whitelist_key_dict(create_store):
    test_columns_to_whitelist = ['metrics.metric0', 'metrics.metric1', 'tags.tag3', 'params.param2']
    actual_col_to_whitelist_dict = create_store._build_columns_to_whitelist_key_dict(
        test_columns_to_whitelist)
    col_to_whitelist_dict = {"metrics": {"metric0", "metric1"},
                             "params": {'param2'}, "tags": {"tag3"}}
    assert actual_col_to_whitelist_dict == col_to_whitelist_dict



@pytest.mark.usefixtures('create_store')
def test__get_orderby_clauses(create_store):
    order_by_list = ['metrics.`metric0` ASC', 'params.`param0` DESC', 'attributes.start_time ASC']
    actual_sort_clauses = create_store._get_orderby_clauses(
        order_by_list=order_by_list)
    sort_clauses = [{'latest_metrics.value': {'order': "asc",
                                              "nested": {"path": "latest_metrics",
                                                         "filter": {"term": {'latest_metrics.key':
                                                                                 "metric0"}}}}},
                    {'params.value': {'order': "desc",
                                      "nested": {"path": "params",
                                                 "filter": {"term": {'params.key': "param0"}}}}},
                    {"start_time": {'order': "asc"}},
                    {"start_time": {'order': "desc"}},
                    {"run_id": {'order': "asc"}}]
    assert actual_sort_clauses == sort_clauses


@mock.patch('mlflow_elasticsearchstore.models.MongoRun.get')
@pytest.mark.usefixtures('create_store')
def test_update_artifacts_location(elastic_run_get_mock, create_store):
    elastic_run_get_mock.return_value = run
    run.update = mock.MagicMock()
    create_store.update_artifacts_location("1", "update_artifacts_location")
    elastic_run_get_mock.assert_called_once_with(id="1")
    run.update.assert_called_once_with(artifact_uri="update_artifacts_location")
