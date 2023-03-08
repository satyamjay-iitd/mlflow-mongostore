import json
import os
import pathlib
import re
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import math
import mlflow
import mlflow.db
import mlflow.store.db.base_sql_model
import pytest
from mlflow import entities
from mlflow.entities import (
    ViewType,
    RunTag,
    SourceType,
    RunStatus,
    Experiment,
    Metric,
    Param,
    ExperimentTag,
)
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_PARAMETER_VALUE,
)
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR
from mlflow.utils import mlflow_tags
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_LOGGED_MODELS
from mlflow.utils.name_utils import _GENERATOR_PREDICATES
from mlflow.utils.os import is_windows
from mlflow.utils.time_utils import get_current_time_millis
from mongoengine.errors import ValidationError

from mlflow_mongostore.models import (
    MongoParam,
    MongoMetric,
    MongoRun,
    MongoExperiment, SequenceId, MongoTag, MongoLatestMetric,
)
from mlflow_mongostore.mongo_store import MongoStore, RunStatusTypes
from tests.store.tracking import AbstractStoreTest

DB_URI = "sqlite:///"
ARTIFACT_URI = "artifact_folder"

pytestmark = pytest.mark.notrackingurimock


class TestMongoStore(unittest.TestCase, AbstractStoreTest):
    def _get_store(self, db_uri="mongodb://localhost:27017/test"):
        return MongoStore(db_uri, ARTIFACT_URI)

    def create_test_run(self):
        return self._run_factory()

    def _setup_db_uri(self):
        if _TRACKING_URI_ENV_VAR in os.environ:
            self.db_url = os.getenv(_TRACKING_URI_ENV_VAR)
        else:
            self.db_url = "mongodb://localhost:27017/test"

    def setUp(self):
        self._setup_db_uri()
        self.store = self._get_store(self.db_url)

    def get_store(self):
        return self.store

    def tearDown(self):
        # Delete all rows in all tables
        for model in (
                MongoRun,
                MongoExperiment,
                SequenceId,
                MongoMetric
        ):
            model.objects().delete()

    def _experiment_factory(self, names):
        if isinstance(names, (list, tuple)):
            ids = []
            for name in names:
                # Sleep to ensure each experiment has a unique creation_time for
                # deterministic experiment search results
                time.sleep(0.001)
                ids.append(self.store.create_experiment(name=name))
            return ids

        time.sleep(0.001)
        return self.store.create_experiment(name=names)

    def test_default_experiment(self):
        experiments = self.store.search_experiments()
        assert len(experiments) == 1

        first = experiments[0]
        assert first.experiment_id == "0"
        assert first.name == "Default"

    def test_default_experiment_lifecycle(self):
        default_experiment = self.store.get_experiment(experiment_id="0")
        assert default_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
        assert default_experiment.lifecycle_stage == entities.LifecycleStage.ACTIVE

        self._experiment_factory("aNothEr")
        all_experiments = [e.name for e in self.store.search_experiments()]
        assert set(all_experiments) == {"aNothEr", "Default"}

        self.store.delete_experiment("0")

        assert [e.name for e in self.store.search_experiments()] == ["aNothEr"]
        another = self.store.get_experiment_by_name(experiment_name="aNothEr")
        assert another.name == "aNothEr"

        default_experiment = self.store.get_experiment(experiment_id="0")
        assert default_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
        assert default_experiment.lifecycle_stage == entities.LifecycleStage.DELETED

        # destroy SqlStore and make a new one
        del self.store
        self.store = self._get_store(self.db_url)

        # test that default experiment is not reactivated
        default_experiment = self.store.get_experiment(experiment_id="0")
        assert default_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
        assert default_experiment.lifecycle_stage == entities.LifecycleStage.DELETED

        assert [e.name for e in self.store.search_experiments()] == ["aNothEr"]
        all_experiments = [e.name for e in self.store.search_experiments(ViewType.ALL)]
        assert set(all_experiments) == {"aNothEr", "Default"}

        # ensure that experiment ID dor active experiment is unchanged
        another = self.store.get_experiment_by_name("aNothEr")
        assert another.name == "aNothEr"

    def test_raise_duplicate_experiments(self):
        with pytest.raises(Exception, match=r"Experiment\(name=.+\) already exists"):
            self._experiment_factory(["test", "test"])

    def test_raise_experiment_dont_exist(self):
        with pytest.raises(Exception, match=r"No Experiment with id=.+ exists"):
            self.store.get_experiment(experiment_id="100")

    def test_delete_experiment(self):
        experiments = self._experiment_factory(["morty", "rick", "rick and morty"])

        all_experiments = self.store.search_experiments()
        assert len(all_experiments) == len(experiments) + 1  # default

        exp_id = experiments[0]
        exp = self.store.get_experiment(exp_id)
        time.sleep(0.01)
        self.store.delete_experiment(exp_id)

        updated_exp = self.store.get_experiment(exp_id)
        assert updated_exp.lifecycle_stage == entities.LifecycleStage.DELETED

        assert len(self.store.search_experiments()) == len(all_experiments) - 1
        assert updated_exp.last_update_time > exp.last_update_time

    def test_get_experiment(self):
        name = "goku"
        experiment_id = self._experiment_factory(name)
        actual = self.store.get_experiment(experiment_id)
        assert actual.name == name
        assert actual.experiment_id == experiment_id

        actual_by_name = self.store.get_experiment_by_name(name)
        assert actual_by_name.name == name
        assert actual_by_name.experiment_id == experiment_id
        assert self.store.get_experiment_by_name("idontexist") is None

    def test_search_experiments_view_type(self):
        experiment_names = ["a", "b"]
        experiment_ids = self._experiment_factory(experiment_names)
        self.store.delete_experiment(experiment_ids[1])

        experiments = self.store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        assert [e.name for e in experiments] == ["a", "Default"]
        experiments = self.store.search_experiments(view_type=ViewType.DELETED_ONLY)
        assert [e.name for e in experiments] == ["b"]
        experiments = self.store.search_experiments(view_type=ViewType.ALL)
        assert [e.name for e in experiments] == ["b", "a", "Default"]

    def test_search_experiments_filter_by_attribute(self):
        experiment_names = ["a", "ab", "Abc"]
        self._experiment_factory(experiment_names)

        experiments = self.store.search_experiments(filter_string="name = 'a'")
        assert [e.name for e in experiments] == ["a"]
        experiments = self.store.search_experiments(filter_string="attribute.name = 'a'")
        assert [e.name for e in experiments] == ["a"]
        experiments = self.store.search_experiments(filter_string="attribute.`name` = 'a'")
        assert [e.name for e in experiments] == ["a"]
        experiments = self.store.search_experiments(filter_string="attribute.`name` != 'a'")
        assert [e.name for e in experiments] == ["Abc", "ab", "Default"]
        experiments = self.store.search_experiments(filter_string="name LIKE 'a%'")
        assert [e.name for e in experiments] == ["ab", "a"]
        experiments = self.store.search_experiments(filter_string="name ILIKE 'a%'")
        assert [e.name for e in experiments] == ["Abc", "ab", "a"]
        experiments = self.store.search_experiments(
            filter_string="name ILIKE 'a%' AND name ILIKE '%b'"
        )
        assert [e.name for e in experiments] == ["ab"]

    def test_search_experiments_filter_by_time_attribute(self):
        # Sleep to ensure that the first experiment has a different creation_time than the default
        # experiment and eliminate flakiness.
        time.sleep(0.001)
        time_before_create1 = get_current_time_millis()
        exp_id1 = self.store.create_experiment("1")
        exp1 = self.store.get_experiment(exp_id1)
        time.sleep(0.001)
        time_before_create2 = get_current_time_millis()
        exp_id2 = self.store.create_experiment("2")
        exp2 = self.store.get_experiment(exp_id2)

        experiments = self.store.search_experiments(
            filter_string=f"creation_time = {exp1.creation_time}"
        )
        assert [e.experiment_id for e in experiments] == [exp_id1]

        experiments = self.store.search_experiments(
            filter_string=f"creation_time != {exp1.creation_time}"
        )
        assert [e.experiment_id for e in experiments] == [exp_id2, self.store.DEFAULT_EXPERIMENT_ID]

        experiments = self.store.search_experiments(
            filter_string=f"creation_time >= {time_before_create1}"
        )
        assert [e.experiment_id for e in experiments] == [exp_id2, exp_id1]

        experiments = self.store.search_experiments(
            filter_string=f"creation_time < {time_before_create2}"
        )
        assert [e.experiment_id for e in experiments] == [exp_id1, self.store.DEFAULT_EXPERIMENT_ID]

        now = get_current_time_millis()
        experiments = self.store.search_experiments(filter_string=f"creation_time >= {now}")
        assert experiments == []

        time.sleep(0.001)
        time_before_rename = get_current_time_millis()
        self.store.rename_experiment(exp_id1, "new_name")
        experiments = self.store.search_experiments(
            filter_string=f"last_update_time >= {time_before_rename}"
        )
        assert [e.experiment_id for e in experiments] == [exp_id1]

        experiments = self.store.search_experiments(
            filter_string=f"last_update_time <= {get_current_time_millis()}"
        )
        assert {e.experiment_id for e in experiments} == {
            exp_id1,
            exp_id2,
            self.store.DEFAULT_EXPERIMENT_ID,
        }

        experiments = self.store.search_experiments(
            filter_string=f"last_update_time = {exp2.last_update_time}"
        )
        assert [e.experiment_id for e in experiments] == [exp_id2]

    def test_search_experiments_filter_by_tag(self):
        experiments = [
            ("exp1", [ExperimentTag("key1", "value"), ExperimentTag("key2", "value")]),
            ("exp2", [ExperimentTag("key1", "vaLue"), ExperimentTag("key2", "vaLue")]),
            ("exp3", [ExperimentTag("k e y 1", "value")]),
        ]
        for name, tags in experiments:
            self.store.create_experiment(name, tags=tags)

        experiments = self.store.search_experiments(filter_string="tag.key1 = 'value'")
        assert [e.name for e in experiments] == ["exp1"]
        experiments = self.store.search_experiments(filter_string="tag.`k e y 1` = 'value'")
        assert [e.name for e in experiments] == ["exp3"]
        experiments = self.store.search_experiments(filter_string="tag.\"k e y 1\" = 'value'")
        assert [e.name for e in experiments] == ["exp3"]
        experiments = self.store.search_experiments(filter_string="tag.key1 != 'value'")
        assert [e.name for e in experiments] == ["exp2"]
        experiments = self.store.search_experiments(filter_string="tag.key1 != 'VALUE'")
        assert [e.name for e in experiments] == ["exp2", "exp1"]
        experiments = self.store.search_experiments(filter_string="tag.key1 LIKE 'val%'")
        assert [e.name for e in experiments] == ["exp1"]
        experiments = self.store.search_experiments(filter_string="tag.key1 LIKE '%Lue'")
        assert [e.name for e in experiments] == ["exp2"]
        experiments = self.store.search_experiments(filter_string="tag.key1 ILIKE '%alu%'")
        assert [e.name for e in experiments] == ["exp2", "exp1"]
        experiments = self.store.search_experiments(
            filter_string="tag.key1 LIKE 'va%' AND tag.key2 LIKE '%Lue'"
        )
        assert [e.name for e in experiments] == ["exp2"]
        experiments = self.store.search_experiments(filter_string="tag.KEY = 'value'")
        assert len(experiments) == 0

    def test_search_experiments_filter_by_attribute_and_tag(self):
        self.store.create_experiment(
            "exp1", tags=[ExperimentTag("a", "1"), ExperimentTag("b", "2")]
        )
        self.store.create_experiment(
            "exp2", tags=[ExperimentTag("a", "3"), ExperimentTag("b", "4")]
        )
        experiments = self.store.search_experiments(
            filter_string="name ILIKE 'exp%' AND tags.a = '1'"
        )
        assert [e.name for e in experiments] == ["exp1"]

    def test_search_experiments_order_by(self):
        experiment_names = ["x", "y", "z"]
        self._experiment_factory(experiment_names)

        experiments = self.store.search_experiments(order_by=["name"])
        assert [e.name for e in experiments] == ["Default", "x", "y", "z"]

        experiments = self.store.search_experiments(order_by=["name ASC"])
        assert [e.name for e in experiments] == ["Default", "x", "y", "z"]

        experiments = self.store.search_experiments(order_by=["name DESC"])
        assert [e.name for e in experiments] == ["z", "y", "x", "Default"]

        experiments = self.store.search_experiments(order_by=["experiment_id DESC"])
        assert [e.name for e in experiments] == ["z", "y", "x", "Default"]

        experiments = self.store.search_experiments(order_by=["name", "experiment_id"])
        assert [e.name for e in experiments] == ["Default", "x", "y", "z"]

    def test_search_experiments_order_by_time_attribute(self):
        # Sleep to ensure that the first experiment has a different creation_time than the default
        # experiment and eliminate flakiness.
        time.sleep(0.001)
        exp_id1 = self.store.create_experiment("1")
        time.sleep(0.001)
        exp_id2 = self.store.create_experiment("2")

        experiments = self.store.search_experiments(order_by=["creation_time"])
        assert [e.experiment_id for e in experiments] == [
            self.store.DEFAULT_EXPERIMENT_ID,
            exp_id1,
            exp_id2,
        ]

        experiments = self.store.search_experiments(order_by=["creation_time DESC"])
        assert [e.experiment_id for e in experiments] == [
            exp_id2,
            exp_id1,
            self.store.DEFAULT_EXPERIMENT_ID,
        ]

        experiments = self.store.search_experiments(order_by=["last_update_time"])
        assert [e.experiment_id for e in experiments] == [
            self.store.DEFAULT_EXPERIMENT_ID,
            exp_id1,
            exp_id2,
        ]

        self.store.rename_experiment(exp_id1, "new_name")
        experiments = self.store.search_experiments(order_by=["last_update_time"])
        assert [e.experiment_id for e in experiments] == [
            self.store.DEFAULT_EXPERIMENT_ID,
            exp_id2,
            exp_id1,
        ]

    def test_search_experiments_max_results(self):
        experiment_names = list(map(str, range(9)))
        self._experiment_factory(experiment_names)
        reversed_experiment_names = experiment_names[::-1]

        experiments = self.store.search_experiments()
        assert [e.name for e in experiments] == reversed_experiment_names + ["Default"]
        experiments = self.store.search_experiments(max_results=3)
        assert [e.name for e in experiments] == reversed_experiment_names[:3]

    def test_search_experiments_max_results_validation(self):
        with pytest.raises(MlflowException, match=r"It must be a positive integer, but got None"):
            self.store.search_experiments(max_results=None)
        with pytest.raises(MlflowException, match=r"It must be a positive integer, but got 0"):
            self.store.search_experiments(max_results=0)
        with pytest.raises(MlflowException, match=r"It must be at most \d+, but got 1000000"):
            self.store.search_experiments(max_results=1_000_000)

    def test_search_experiments_pagination(self):
        experiment_names = list(map(str, range(9)))
        self._experiment_factory(experiment_names)
        reversed_experiment_names = experiment_names[::-1]

        experiments = self.store.search_experiments(max_results=4)
        assert [e.name for e in experiments] == reversed_experiment_names[:4]
        assert experiments.token is not None

        experiments = self.store.search_experiments(max_results=4, page_token=experiments.token)
        assert [e.name for e in experiments] == reversed_experiment_names[4:8]
        assert experiments.token is not None

        experiments = self.store.search_experiments(max_results=4, page_token=experiments.token)
        assert [e.name for e in experiments] == reversed_experiment_names[8:] + ["Default"]
        assert experiments.token is None

    def test_create_experiments(self):
        result = MongoExperiment.objects()
        assert len(result) == 1
        time_before_create = get_current_time_millis()
        experiment_id = self.store.create_experiment(name="test exp")
        assert experiment_id == "1"
        result = MongoExperiment.objects()
        assert len(result) == 2

        test_exp = MongoExperiment.objects(name="test exp")[0]
        assert str(test_exp.exp_id) == experiment_id
        assert test_exp.name == "test exp"

        actual = self.store.get_experiment(experiment_id)
        assert actual.experiment_id == experiment_id
        assert actual.name == "test exp"
        assert actual.creation_time >= time_before_create
        assert actual.last_update_time == actual.creation_time

    def test_create_experiment_with_tags_works_correctly(self):
        experiment_id = self.store.create_experiment(
            name="test exp",
            artifact_location="some location",
            tags=[ExperimentTag("key1", "val1"), ExperimentTag("key2", "val2")],
        )
        experiment = self.store.get_experiment(experiment_id)
        assert len(experiment.tags) == 2
        assert experiment.tags["key1"] == "val1"
        assert experiment.tags["key2"] == "val2"

    def test_set_experiment_tag(self):
        exp_id = self._experiment_factory("setExperimentTagExp")
        tag = entities.ExperimentTag("tag0", "value0")
        new_tag = entities.RunTag("tag0", "value00000")
        self.store.set_experiment_tag(exp_id, tag)
        experiment = self.store.get_experiment(exp_id)
        assert experiment.tags["tag0"] == "value0"
        # test that updating a tag works
        self.store.set_experiment_tag(exp_id, new_tag)
        experiment = self.store.get_experiment(exp_id)
        assert experiment.tags["tag0"] == "value00000"
        # test that setting a tag on 1 experiment does not impact another experiment.
        exp_id_2 = self._experiment_factory("setExperimentTagExp2")
        experiment2 = self.store.get_experiment(exp_id_2)
        assert len(experiment2.tags) == 0
        # setting a tag on different experiments maintains different values across experiments
        different_tag = entities.RunTag("tag0", "differentValue")
        self.store.set_experiment_tag(exp_id_2, different_tag)
        experiment = self.store.get_experiment(exp_id)
        assert experiment.tags["tag0"] == "value00000"
        experiment2 = self.store.get_experiment(exp_id_2)
        assert experiment2.tags["tag0"] == "differentValue"
        # test can set multi-line tags
        multi_line_Tag = entities.ExperimentTag("multiline tag", "value2\nvalue2\nvalue2")
        self.store.set_experiment_tag(exp_id, multi_line_Tag)
        experiment = self.store.get_experiment(exp_id)
        assert experiment.tags["multiline tag"] == "value2\nvalue2\nvalue2"
        # test cannot set tags that are too long
        long_tag = entities.ExperimentTag("longTagKey", "a" * 5001)
        with pytest.raises(MlflowException, match="exceeded length limit of 5000"):
            self.store.set_experiment_tag(exp_id, long_tag)
        # test can set tags that are somewhat long
        long_tag = entities.ExperimentTag("longTagKey", "a" * 4999)
        self.store.set_experiment_tag(exp_id, long_tag)
        # test cannot set tags on deleted experiments
        self.store.delete_experiment(exp_id)
        with pytest.raises(MlflowException, match="must be in the 'active' state"):
            self.store.set_experiment_tag(exp_id, entities.ExperimentTag("should", "notset"))

    def test_rename_experiment(self):
        new_name = "new name"
        experiment_id = self._experiment_factory("test name")
        experiment = self.store.get_experiment(experiment_id)
        time.sleep(0.01)
        self.store.rename_experiment(experiment_id, new_name)

        renamed_experiment = self.store.get_experiment(experiment_id)

        assert renamed_experiment.name == new_name
        assert renamed_experiment.last_update_time > experiment.last_update_time

    def test_restore_experiment(self):
        experiment_id = self._experiment_factory("helloexp")
        exp = self.store.get_experiment(experiment_id)
        assert exp.lifecycle_stage == entities.LifecycleStage.ACTIVE

        experiment_id = exp.experiment_id
        self.store.delete_experiment(experiment_id)

        deleted = self.store.get_experiment(experiment_id)
        assert deleted.experiment_id == experiment_id
        assert deleted.lifecycle_stage == entities.LifecycleStage.DELETED
        time.sleep(0.01)
        self.store.restore_experiment(exp.experiment_id)
        restored = self.store.get_experiment(exp.experiment_id)
        assert restored.experiment_id == experiment_id
        assert restored.lifecycle_stage == entities.LifecycleStage.ACTIVE
        assert restored.last_update_time > deleted.last_update_time

    def _get_run_configs(self, experiment_id=None, tags=None, start_time=None):
        return {
            "experiment_id": experiment_id,
            "user_id": "Anderson",
            "start_time": start_time if start_time is not None else get_current_time_millis(),
            "tags": tags,
            "run_name": "name",
        }

    def _run_factory(self, config=None):
        if not config:
            config = self._get_run_configs()

        experiment_id = config.get("experiment_id", None)
        if not experiment_id:
            experiment_id = self._experiment_factory("test exp")
            config["experiment_id"] = experiment_id

        return self.store.create_run(**config)

    def test_delete_restore_experiment_with_runs(self):
        experiment_id = self._experiment_factory("test exp")
        run1 = self._run_factory(config=self._get_run_configs(experiment_id)).info.run_id
        run2 = self._run_factory(config=self._get_run_configs(experiment_id)).info.run_id
        self.store.delete_run(run1)
        run_ids = [run1, run2]

        self.store.delete_experiment(experiment_id)

        updated_exp = self.store.get_experiment(experiment_id)
        assert updated_exp.lifecycle_stage == entities.LifecycleStage.DELETED

        deleted_run_list = self.store.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.DELETED_ONLY,
        )

        assert len(deleted_run_list) == 2
        for deleted_run in deleted_run_list:
            assert deleted_run.info.lifecycle_stage == entities.LifecycleStage.DELETED
            assert deleted_run.info.experiment_id in experiment_id
            assert deleted_run.info.run_id in run_ids
            assert (
                    self.store._get_run(deleted_run.info.run_id).deleted_time is not None
            )

        self.store.restore_experiment(experiment_id)

        updated_exp = self.store.get_experiment(experiment_id)
        assert updated_exp.lifecycle_stage == entities.LifecycleStage.ACTIVE

        restored_run_list = self.store.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
        )

        assert len(restored_run_list) == 2
        for restored_run in restored_run_list:
            assert restored_run.info.lifecycle_stage == entities.LifecycleStage.ACTIVE
            assert self.store._get_run(restored_run.info.run_id).deleted_time is None
            assert restored_run.info.experiment_id in experiment_id
            assert restored_run.info.run_id in run_ids

    def test_run_needs_uuid(self):
        # Depending on the implementation, a NULL identity key may result in different
        # exceptions, including IntegrityError (sqlite) and FlushError (MysQL).
        # Therefore, we check for the more generic 'SQLAlchemyError'
        with pytest.raises(ValidationError) as exception_context:
            run = MongoRun()
            run.save()
        assert exception_context.value.errors.get('run_uuid') is not None

    def test_run_data_model(self):
        run_id = uuid.uuid4().hex
        m1 = MongoMetric(key="accuracy", value=0.89, run_uuid=run_id)
        m2 = MongoMetric(key="recall", value=0.89, run_uuid=run_id)
        p1 = MongoParam(key="loss", value="test param")
        p2 = MongoParam(key="blue", value="test param")

        run_data = MongoRun(run_uuid=run_id, params=[p1, p2])
        run_data.save()
        m1.save()
        m2.save()

        run_datums = MongoRun.objects()

        actual = run_datums[0]
        assert len(run_datums) == 1
        assert len(actual.params) == 2
        assert len(actual.metrics) == 2

    def test_run_info(self):
        experiment_id = self._experiment_factory("test exp")
        config = {
            "experiment_id": experiment_id,
            "name": "test run",
            "user_id": "Anderson",
            "run_uuid": "test",
            "status": RunStatus.to_string(RunStatus.SCHEDULED),
            "source_type": SourceType.to_string(SourceType.LOCAL),
            "source_name": "Python application",
            "entry_point_name": "main.py",
            "start_time": get_current_time_millis(),
            "end_time": get_current_time_millis(),
            "source_version": mlflow.__version__,
            "lifecycle_stage": entities.LifecycleStage.ACTIVE,
            "artifact_uri": "//",
        }
        run = MongoRun(**config).to_mlflow_entity()

        for k, v in config.items():
            # These keys were removed from RunInfo.
            if k in ["source_name", "source_type", "source_version", "name", "entry_point_name"]:
                continue

            v2 = getattr(run.info, k)
            if k == "source_type":
                assert v == SourceType.to_string(v2)
            else:
                assert v == v2

    def test_create_run_with_tags(self):
        experiment_id = self._experiment_factory("test_create_run")
        tags = [RunTag("3", "4"), RunTag("1", "2")]
        expected = self._get_run_configs(experiment_id=experiment_id, tags=tags)

        actual = self.store.create_run(**expected)

        assert actual.info.experiment_id == experiment_id
        assert actual.info.user_id == expected["user_id"]
        assert actual.info.run_name == expected["run_name"]
        assert actual.info.start_time == expected["start_time"]

        assert len(actual.data.tags) == len(tags)
        expected_tags = {tag.key: tag.value for tag in tags}
        assert actual.data.tags == expected_tags

    def test_create_run_sets_name(self):
        experiment_id = self._experiment_factory("test_create_run_run_name")
        configs = self._get_run_configs(experiment_id=experiment_id)
        run_id = self.store.create_run(**configs).info.run_id
        run = self.store.get_run(run_id)
        assert run.info.run_name == configs["run_name"]
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == configs["run_name"]
        run_id = self.store.create_run(
            experiment_id=experiment_id,
            user_id="user",
            start_time=0,
            run_name=None,
            tags=[RunTag(mlflow_tags.MLFLOW_RUN_NAME, "test")],
        ).info.run_id
        run = self.store.get_run(run_id)
        assert run.info.run_name == "test"

        with pytest.raises(
                MlflowException,
                match=re.escape(
                    "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                    "different values (run_name='test', mlflow.runName='test_2').",
                ),
        ):
            self.store.create_run(
                experiment_id=experiment_id,
                user_id="user",
                start_time=0,
                run_name="test",
                tags=[RunTag(mlflow_tags.MLFLOW_RUN_NAME, "test_2")],
            )

    def test_get_run_with_name(self):
        experiment_id = self._experiment_factory("test_get_run")
        configs = self._get_run_configs(experiment_id=experiment_id)

        run_id = self.store.create_run(**configs).info.run_id

        run = self.store.get_run(run_id)

        assert run.info.experiment_id == experiment_id
        assert run.info.run_name == configs["run_name"]

        no_run_configs = {
            "experiment_id": experiment_id,
            "user_id": "Anderson",
            "start_time": get_current_time_millis(),
            "tags": [],
            "run_name": None,
        }
        run_id = self.store.create_run(**no_run_configs).info.run_id
        run = self.store.get_run(run_id)
        assert run.info.run_name.split("-")[0] in _GENERATOR_PREDICATES

        name_empty_str_run = self.store.create_run(**{**configs, **{"run_name": ""}})
        run_name = name_empty_str_run.info.run_name
        assert run_name.split("-")[0] in _GENERATOR_PREDICATES

    def test_to_mlflow_entity_and_proto(self):
        # Create a run and log metrics, params, tags to the run
        created_run = self._run_factory()
        run_id = created_run.info.run_id
        self.store.log_metric(
            run_id=run_id, metric=entities.Metric(key="my-metric", value=3.4, timestamp=0, step=0)
        )
        self.store.log_param(run_id=run_id, param=Param(key="my-param", value="param-val"))
        self.store.set_tag(run_id=run_id, tag=RunTag(key="my-tag", value="tag-val"))

        # Verify that we can fetch the run & convert it to proto - Python protobuf bindings
        # will perform type-checking to ensure all values have the right types
        run = self.store.get_run(run_id)
        run.to_proto()

        # Verify attributes of the Python run entity
        assert isinstance(run.info, entities.RunInfo)
        assert isinstance(run.data, entities.RunData)

        assert run.data.metrics == {"my-metric": 3.4}
        assert run.data.params == {"my-param": "param-val"}
        assert run.data.tags["my-tag"] == "tag-val"

        # Get the parent experiment of the run, verify it can be converted to protobuf
        exp = self.store.get_experiment(run.info.experiment_id)
        exp.to_proto()

    def test_delete_run(self):
        run = self._run_factory()

        self.store.delete_run(run.info.run_id)

        actual = MongoRun.objects(run_uuid=run.info.run_id)[0]
        assert actual.lifecycle_stage == entities.LifecycleStage.DELETED
        assert (
                actual.deleted_time is not None
        )  # deleted time should be updated and thus not None anymore

        deleted_run = self.store.get_run(run.info.run_id)
        assert actual.run_uuid == deleted_run.info.run_id

    def test_hard_delete_run(self):
        run = self._run_factory()
        metric = entities.Metric("blahmetric", 100.0, get_current_time_millis(), 0)
        self.store.log_metric(run.info.run_id, metric)
        param = entities.Param("blahparam", "100.0")
        self.store.log_param(run.info.run_id, param)
        tag = entities.RunTag("test tag", "a boogie")
        self.store.set_tag(run.info.run_id, tag)

        self.store._hard_delete_run(run.info.run_id)

        actual_run = MongoRun.objects(run_uuid=run.info.run_id)
        assert len(actual_run) == 0

    def test_get_deleted_runs(self):
        run = self._run_factory()
        deleted_run_ids = self.store._get_deleted_runs()
        assert deleted_run_ids == []

        self.store.delete_run(run.info.run_uuid)
        deleted_run_ids = self.store._get_deleted_runs()
        assert deleted_run_ids == [run.info.run_uuid]

    def test_log_metric(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = 100.0
        metric = entities.Metric(tkey, tval, get_current_time_millis(), 0)
        metric2 = entities.Metric(tkey, tval, get_current_time_millis() + 2, 0)
        nan_metric = entities.Metric("NaN", float("nan"), 0, 0)
        pos_inf_metric = entities.Metric("PosInf", float("inf"), 0, 0)
        neg_inf_metric = entities.Metric("NegInf", -float("inf"), 0, 0)
        self.store.log_metric(run.info.run_id, metric)
        self.store.log_metric(run.info.run_id, metric2)
        self.store.log_metric(run.info.run_id, nan_metric)
        self.store.log_metric(run.info.run_id, pos_inf_metric)
        self.store.log_metric(run.info.run_id, neg_inf_metric)

        run = self.store.get_run(run.info.run_id)
        assert tkey in run.data.metrics and run.data.metrics[tkey] == tval

        # SQL store _get_run method returns full history of recorded metrics.
        # Should return duplicates as well
        # MLflow RunData contains only the last reported values for metrics.
        sql_run_metrics = self.store._get_run(run.info.run_id).metrics
        assert len(sql_run_metrics) == 5
        assert len(run.data.metrics) == 4
        assert math.isnan(run.data.metrics["NaN"])
        assert run.data.metrics["PosInf"] == 1.7976931348623157e308
        assert run.data.metrics["NegInf"] == -1.7976931348623157e308

    def test_log_metric_concurrent_logging_succeeds(self):
        """
        Verifies that concurrent logging succeeds without deadlock, which has been an issue
        in previous MLFlow releases
        """
        experiment_id = self._experiment_factory("concurrency_exp")
        run_config = self._get_run_configs(experiment_id=experiment_id)
        run1 = self._run_factory(run_config)
        run2 = self._run_factory(run_config)

        def log_metrics(run):
            for metric_val in range(100):
                self.store.log_metric(
                    run.info.run_id, Metric("metric_key", metric_val, get_current_time_millis(), 0)
                )
            for batch_idx in range(5):
                self.store.log_batch(
                    run.info.run_id,
                    metrics=[
                        Metric(
                            f"metric_batch_{batch_idx}",
                            (batch_idx * 100) + val_offset,
                            get_current_time_millis(),
                            0,
                        )
                        for val_offset in range(100)
                    ],
                    params=[],
                    tags=[],
                )
            for metric_val in range(100):
                self.store.log_metric(
                    run.info.run_id, Metric("metric_key", metric_val, get_current_time_millis(), 0)
                )
            return "success"

        log_metrics_futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Log metrics to two runs across four threads
            log_metrics_futures = [
                executor.submit(log_metrics, run) for run in [run1, run2, run1, run2]
            ]

        for future in log_metrics_futures:
            assert future.result() == "success"

        for run in [run1, run2, run1, run2]:
            # We visit each run twice, logging 100 metric entries for 6 metric names; the same entry
            # may be written multiple times concurrently; we assert that at least 100 metric entries
            # are present because at least 100 unique entries must have been written
            assert len(self.store.get_metric_history(run.info.run_id, "metric_key")) >= 100
            for batch_idx in range(5):
                assert (
                        len(self.store.get_metric_history(run.info.run_id, f"metric_batch_{batch_idx}"))
                        >= 100
                )

    def test_log_metric_allows_multiple_values_at_same_ts_and_run_data_uses_max_ts_value(self):
        run = self._run_factory()
        run_id = run.info.run_id
        metric_name = "test-metric-1"
        # Check that we get the max of (step, timestamp, value) in that order
        tuples_to_log = [
            (0, 100, 1000),
            (3, 40, 100),  # larger step wins even though it has smaller value
            (3, 50, 10),  # larger timestamp wins even though it has smaller value
            (3, 50, 20),  # tiebreak by max value
            (3, 50, 20),  # duplicate metrics with same (step, timestamp, value) are ok
            # verify that we can log steps out of order / negative steps
            (-3, 900, 900),
            (-1, 800, 800),
        ]
        for step, timestamp, value in reversed(tuples_to_log):
            self.store.log_metric(run_id, Metric(metric_name, value, timestamp, step))

        metric_history = self.store.get_metric_history(run_id, metric_name)
        logged_tuples = [(m.step, m.timestamp, m.value) for m in metric_history]
        assert set(logged_tuples) == set(tuples_to_log)

        run_data = self.store.get_run(run_id).data
        run_metrics = run_data.metrics
        assert len(run_metrics) == 1
        assert run_metrics[metric_name] == 20
        metric_obj = run_data._metric_objs[0]
        assert metric_obj.key == metric_name
        assert metric_obj.step == 3
        assert metric_obj.timestamp == 50
        assert metric_obj.value == 20

    def test_get_metric_history_paginated_request_raises(self):
        with pytest.raises(
                MlflowException,
                match="The SQLAlchemyStore backend does not support pagination for the "
                      "`get_metric_history` API.",
        ):
            self.store.get_metric_history(
                "fake_run", "fake_metric", max_results=50, page_token="42"
            )

    def test_log_null_metric(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = None
        metric = entities.Metric(tkey, tval, get_current_time_millis(), 0)

        with pytest.raises(
                MlflowException, match=r"Got invalid value None for metric"
        ) as exception_context:
            self.store.log_metric(run.info.run_id, metric)
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_log_param(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = "100.0"
        param = entities.Param(tkey, tval)
        param2 = entities.Param("new param", "new key")
        self.store.log_param(run.info.run_id, param)
        self.store.log_param(run.info.run_id, param2)
        self.store.log_param(run.info.run_id, param2)

        run = self.store.get_run(run.info.run_id)
        assert len(run.data.params) == 2
        assert tkey in run.data.params and run.data.params[tkey] == tval

    def test_log_param_uniqueness(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = "100.0"
        param = entities.Param(tkey, tval)
        param2 = entities.Param(tkey, "newval")
        self.store.log_param(run.info.run_id, param)

        with pytest.raises(MlflowException, match=r"Changing param values is not allowed"):
            self.store.log_param(run.info.run_id, param2)

    def test_log_empty_str(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = ""
        param = entities.Param(tkey, tval)
        param2 = entities.Param("new param", "new key")
        self.store.log_param(run.info.run_id, param)
        self.store.log_param(run.info.run_id, param2)

        run = self.store.get_run(run.info.run_id)
        assert len(run.data.params) == 2
        assert tkey in run.data.params and run.data.params[tkey] == tval

    def test_log_null_param(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = None
        param = entities.Param(tkey, tval)

        with pytest.raises(MlflowException) as exception_context:
            self.store.log_param(run.info.run_id, param)

    def test_log_param_max_length_value(self):
        run = self._run_factory()
        tkey = "blahmetric"
        tval = "x" * 500
        param = entities.Param(tkey, tval)
        self.store.log_param(run.info.run_id, param)
        run = self.store.get_run(run.info.run_id)
        assert run.data.params[tkey] == str(tval)
        with pytest.raises(MlflowException, match="exceeded length"):
            self.store.log_param(run.info.run_id, entities.Param(tkey, "x" * 1000))

    def test_set_tag(self):
        run = self._run_factory()

        tkey = "test tag"
        tval = "a boogie"
        new_val = "new val"
        tag = entities.RunTag(tkey, tval)
        new_tag = entities.RunTag(tkey, new_val)
        self.store.set_tag(run.info.run_id, tag)
        # Overwriting tags is allowed
        self.store.set_tag(run.info.run_id, new_tag)
        # test setting tags that are too long fails.
        with pytest.raises(MlflowException, match="exceeded length limit of 5000"):
            self.store.set_tag(run.info.run_id, entities.RunTag("longTagKey", "a" * 5001))
        # test can set tags that are somewhat long
        self.store.set_tag(run.info.run_id, entities.RunTag("longTagKey", "a" * 4999))
        run = self.store.get_run(run.info.run_id)
        assert tkey in run.data.tags and run.data.tags[tkey] == new_val

    def test_delete_tag(self):
        run = self._run_factory()
        k0, v0 = "tag0", "val0"
        k1, v1 = "tag1", "val1"
        tag0 = entities.RunTag(k0, v0)
        tag1 = entities.RunTag(k1, v1)
        self.store.set_tag(run.info.run_id, tag0)
        self.store.set_tag(run.info.run_id, tag1)
        # delete a tag and check whether it is correctly deleted.
        self.store.delete_tag(run.info.run_id, k0)
        run = self.store.get_run(run.info.run_id)
        assert k0 not in run.data.tags
        assert k1 in run.data.tags and run.data.tags[k1] == v1

        # test that deleting a tag works correctly with multiple runs having the same tag.
        run2 = self._run_factory(config=self._get_run_configs(run.info.experiment_id))
        self.store.set_tag(run.info.run_id, tag0)
        self.store.set_tag(run2.info.run_id, tag0)
        self.store.delete_tag(run.info.run_id, k0)
        run = self.store.get_run(run.info.run_id)
        run2 = self.store.get_run(run2.info.run_id)
        assert k0 not in run.data.tags
        assert k0 in run2.data.tags
        # test that you cannot delete tags that don't exist.
        with pytest.raises(MlflowException, match="No tag with name"):
            self.store.delete_tag(run.info.run_id, "fakeTag")
        # test that you cannot delete tags for nonexistent runs
        with pytest.raises(MlflowException, match="Run with id=randomRunId not found"):
            self.store.delete_tag("randomRunId", k0)
        # test that you cannot delete tags for deleted runs.
        self.store.delete_run(run.info.run_id)
        with pytest.raises(MlflowException, match="must be in the 'active' state"):
            self.store.delete_tag(run.info.run_id, k1)

    def test_get_metric_history(self):
        run = self._run_factory()

        key = "test"
        expected = [
            MongoMetric(key=key, value=0.6, timestamp=1, step=0).to_mlflow_entity(),
            MongoMetric(key=key, value=0.7, timestamp=2, step=0).to_mlflow_entity(),
        ]

        for metric in expected:
            self.store.log_metric(run.info.run_id, metric)

        actual = self.store.get_metric_history(run.info.run_id, key)

        assert sorted(
            [(m.key, m.value, m.timestamp) for m in expected],
        ) == sorted(
            [(m.key, m.value, m.timestamp) for m in actual],
        )

    def test_update_run_info(self):
        experiment_id = self._experiment_factory("test_update_run_info")
        for new_status_string in RunStatusTypes:
            run = self._run_factory(config=self._get_run_configs(experiment_id=experiment_id))
            endtime = get_current_time_millis()
            actual = self.store.update_run_info(
                run.info.run_id, RunStatus.from_string(new_status_string), endtime, None
            )
            assert actual.status == new_status_string
            assert actual.end_time == endtime

    def test_update_run_name(self):
        experiment_id = self._experiment_factory("test_update_run_name")
        configs = self._get_run_configs(experiment_id=experiment_id)

        run_id = self.store.create_run(**configs).info.run_id
        run = self.store.get_run(run_id)
        assert run.info.run_name == configs["run_name"]

        self.store.update_run_info(run_id, RunStatus.FINISHED, 1000, "new name")
        run = self.store.get_run(run_id)
        assert run.info.run_name == "new name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "new name"

        self.store.update_run_info(run_id, RunStatus.FINISHED, 1000, None)
        run = self.store.get_run(run_id)
        assert run.info.run_name == "new name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "new name"

        self.store.update_run_info(run_id, RunStatus.FINISHED, 1000, "")
        run = self.store.get_run(run_id)
        assert run.info.run_name == "new name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "new name"

        self.store.delete_tag(run_id, mlflow_tags.MLFLOW_RUN_NAME)
        run = self.store.get_run(run_id)
        assert run.info.run_name == "new name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) is None

        self.store.update_run_info(run_id, RunStatus.FINISHED, 1000, "newer name")
        run = self.store.get_run(run_id)
        assert run.info.run_name == "newer name"
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "newer name"

        self.store.set_tag(run_id, entities.RunTag(mlflow_tags.MLFLOW_RUN_NAME, "newest name"))
        run = self.store.get_run(run_id)
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "newest name"
        assert run.info.run_name == "newest name"

        self.store.log_batch(
            run_id,
            metrics=[],
            params=[],
            tags=[entities.RunTag(mlflow_tags.MLFLOW_RUN_NAME, "batch name")],
        )
        run = self.store.get_run(run_id)
        assert run.data.tags.get(mlflow_tags.MLFLOW_RUN_NAME) == "batch name"
        assert run.info.run_name == "batch name"

    def test_delete_restore_run(self):
        run = self._run_factory()
        assert run.info.lifecycle_stage == entities.LifecycleStage.ACTIVE

        # Verify that active runs can be restored (run restoration is idempotent)
        self.store.restore_run(run.info.run_id)

        # Verify that run deletion is idempotent
        self.store.delete_run(run.info.run_id)
        self.store.delete_run(run.info.run_id)

        deleted = self.store.get_run(run.info.run_id)
        assert deleted.info.run_id == run.info.run_id
        assert deleted.info.lifecycle_stage == entities.LifecycleStage.DELETED
        assert self.store._get_run(deleted.info.run_id).deleted_time is not None
        # Verify that restoration of a deleted run is idempotent
        self.store.restore_run(run.info.run_id)
        self.store.restore_run(run.info.run_id)
        restored = self.store.get_run(run.info.run_id)
        assert restored.info.run_id == run.info.run_id
        assert restored.info.lifecycle_stage == entities.LifecycleStage.ACTIVE
        assert self.store._get_run(restored.info.run_id).deleted_time is None

    def test_error_logging_to_deleted_run(self):
        exp = self._experiment_factory("error_logging")
        run_id = self._run_factory(self._get_run_configs(experiment_id=exp)).info.run_id

        self.store.delete_run(run_id)
        assert self.store.get_run(run_id).info.lifecycle_stage == entities.LifecycleStage.DELETED
        with pytest.raises(MlflowException, match=r"The run .+ must be in the 'active' state"):
            self.store.log_param(run_id, entities.Param("p1345", "v1"))

        with pytest.raises(MlflowException, match=r"The run .+ must be in the 'active' state"):
            self.store.log_metric(run_id, entities.Metric("m1345", 1.0, 123, 0))

        with pytest.raises(MlflowException, match=r"The run .+ must be in the 'active' state"):
            self.store.set_tag(run_id, entities.RunTag("t1345", "tv1"))

        # restore this run and try again
        self.store.restore_run(run_id)
        assert self.store.get_run(run_id).info.lifecycle_stage == entities.LifecycleStage.ACTIVE
        self.store.log_param(run_id, entities.Param("p1345", "v22"))
        self.store.log_metric(run_id, entities.Metric("m1345", 34.0, 85, 1))  # earlier timestamp
        self.store.set_tag(run_id, entities.RunTag("t1345", "tv44"))

        run = self.store.get_run(run_id)
        assert run.data.params == {"p1345": "v22"}
        assert run.data.metrics == {"m1345": 34.0}
        metric_history = self.store.get_metric_history(run_id, "m1345")
        assert len(metric_history) == 1
        metric_obj = metric_history[0]
        assert metric_obj.key == "m1345"
        assert metric_obj.value == 34.0
        assert metric_obj.timestamp == 85
        assert metric_obj.step == 1
        assert {("t1345", "tv44")} <= set(run.data.tags.items())

    # Tests for Search API
    def _search(
            self,
            experiment_id,
            filter_string=None,
            run_view_type=ViewType.ALL,
            max_results=SEARCH_MAX_RESULTS_DEFAULT,
    ):
        exps = [experiment_id] if isinstance(experiment_id, str) else experiment_id
        return [
            r.info.run_id
            for r in self.store.search_runs(exps, filter_string, run_view_type, max_results)
        ]

    def get_ordered_runs(self, order_clauses, experiment_id):
        return [
            r.data.tags[mlflow_tags.MLFLOW_RUN_NAME]
            for r in self.store.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                run_view_type=ViewType.ALL,
                order_by=order_clauses,
            )
        ]

    def test_order_by_metric_tag_param(self):
        experiment_id = self.store.create_experiment("order_by_metric")

        def create_and_log_run(names):
            name = str(names[0]) + "/" + names[1]
            run_id = self.store.create_run(
                experiment_id,
                user_id="MrDuck",
                start_time=123,
                tags=[entities.RunTag("metric", names[1])],
                run_name=name,
            ).info.run_id
            if names[0] is not None:
                self.store.log_metric(run_id, entities.Metric("x", float(names[0]), 1, 0))
                self.store.log_metric(run_id, entities.Metric("y", float(names[1]), 1, 0))
            self.store.log_param(run_id, entities.Param("metric", names[1]))
            return run_id

        # the expected order in ascending sort is :
        # inf > number > -inf > None > nan
        for names in zip(
                [None, "nan", "inf", "-inf", "-1000", "0", "0", "1000"],
                ["1", "2", "3", "4", "5", "6", "7", "8"],
        ):
            create_and_log_run(names)

        # asc/asc
        assert self.get_ordered_runs(["metrics.x asc", "metrics.y asc"], experiment_id) == [
            "-inf/4",
            "-1000/5",
            "0/6",
            "0/7",
            "1000/8",
            "inf/3",
            "nan/2",
            "None/1",
        ]

        assert self.get_ordered_runs(["metrics.x asc", "tag.metric asc"], experiment_id) == [
            "-inf/4",
            "-1000/5",
            "0/6",
            "0/7",
            "1000/8",
            "inf/3",
            "nan/2",
            "None/1",
        ]

        # asc/desc
        assert self.get_ordered_runs(["metrics.x asc", "metrics.y desc"], experiment_id) == [
            "-inf/4",
            "-1000/5",
            "0/7",
            "0/6",
            "1000/8",
            "inf/3",
            "nan/2",
            "None/1",
        ]

        assert self.get_ordered_runs(["metrics.x asc", "tag.metric desc"], experiment_id) == [
            "-inf/4",
            "-1000/5",
            "0/7",
            "0/6",
            "1000/8",
            "inf/3",
            "nan/2",
            "None/1",
        ]

        # desc / asc
        assert self.get_ordered_runs(["metrics.x desc", "metrics.y asc"], experiment_id) == [
            "inf/3",
            "1000/8",
            "0/6",
            "0/7",
            "-1000/5",
            "-inf/4",
            "nan/2",
            "None/1",
        ]

        # desc / desc
        assert self.get_ordered_runs(["metrics.x desc", "param.metric desc"], experiment_id) == [
            "inf/3",
            "1000/8",
            "0/7",
            "0/6",
            "-1000/5",
            "-inf/4",
            "nan/2",
            "None/1",
        ]
    
    def test_order_by_attributes(self):
        experiment_id = self.store.create_experiment("order_by_attributes")

        def create_run(start_time, end):
            return self.store.create_run(
                experiment_id,
                user_id="MrDuck",
                start_time=start_time,
                tags=[],
                run_name=str(end),
            ).info.run_id

        start_time = 123
        for end in [234, None, 456, -123, 789, 123]:
            run_id = create_run(start_time, end)
            self.store.update_run_info(
                run_id, run_status=RunStatus.FINISHED, end_time=end, run_name=None
            )
            start_time += 1

        # asc
        assert self.get_ordered_runs(["attribute.end_time asc"], experiment_id) == [
            "-123",
            "123",
            "234",
            "456",
            "789",
            "None",
        ]

        # desc
        assert self.get_ordered_runs(["attribute.end_time desc"], experiment_id) == [
            "789",
            "456",
            "234",
            "123",
            "-123",
            "None",
        ]

        # Sort priority correctly handled
        assert self.get_ordered_runs(
            ["attribute.start_time asc", "attribute.end_time desc"], experiment_id
        ) == ["234", "None", "456", "-123", "789", "123"]

    def test_search_vanilla(self):
        exp = self._experiment_factory("search_vanilla")
        runs = [self._run_factory(self._get_run_configs(exp)).info.run_id for r in range(3)]

        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ALL))
        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        assert self._search(exp, run_view_type=ViewType.DELETED_ONLY) == []

        first = runs[0]

        self.store.delete_run(first)
        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ALL))
        assert sorted(
            runs[1:],
        ) == sorted(self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        assert self._search(exp, run_view_type=ViewType.DELETED_ONLY) == [first]

        self.store.restore_run(first)
        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ALL))
        assert sorted(
            runs,
        ) == sorted(self._search(exp, run_view_type=ViewType.ACTIVE_ONLY))
        assert self._search(exp, run_view_type=ViewType.DELETED_ONLY) == []

    def test_search_params(self):
        experiment_id = self._experiment_factory("search_params")
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.log_param(r1, entities.Param("generic_param", "p_val"))
        self.store.log_param(r2, entities.Param("generic_param", "p_val"))

        self.store.log_param(r1, entities.Param("generic_2", "some value"))
        self.store.log_param(r2, entities.Param("generic_2", "another value"))

        self.store.log_param(r1, entities.Param("p_a", "abc"))
        self.store.log_param(r2, entities.Param("p_b", "ABC"))

        # test search returns both runs
        filter_string = "params.generic_param = 'p_val'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        # test search returns appropriate run (same key different values per run)
        filter_string = "params.generic_2 = 'some value'"
        assert self._search(experiment_id, filter_string) == [r1]
        filter_string = "params.generic_2 = 'another value'"
        assert self._search(experiment_id, filter_string) == [r2]

        filter_string = "params.generic_param = 'wrong_val'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_param != 'p_val'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_param != 'wrong_val'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))
        filter_string = "params.generic_2 != 'wrong_val'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "params.p_a = 'abc'"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "params.p_a = 'ABC'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.p_a != 'ABC'"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "params.p_b = 'ABC'"
        assert self._search(experiment_id, filter_string) == [r2]

        filter_string = "params.generic_2 LIKE '%other%'"
        assert self._search(experiment_id, filter_string) == [r2]

        filter_string = "params.generic_2 LIKE 'other%'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_2 LIKE '%other'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_2 LIKE 'other'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_2 LIKE '%Other%'"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "params.generic_2 ILIKE '%Other%'"
        assert self._search(experiment_id, filter_string) == [r2]

    def test_search_tags(self):
        experiment_id = self._experiment_factory("search_tags")
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.set_tag(r1, entities.RunTag("generic_tag", "p_val"))
        self.store.set_tag(r2, entities.RunTag("generic_tag", "p_val"))

        self.store.set_tag(r1, entities.RunTag("generic_2", "some value"))
        self.store.set_tag(r2, entities.RunTag("generic_2", "another value"))

        self.store.set_tag(r1, entities.RunTag("p_a", "abc"))
        self.store.set_tag(r2, entities.RunTag("p_b", "ABC"))

        # test search returns both runs
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string="tags.generic_tag = 'p_val'"))
        assert self._search(experiment_id, filter_string="tags.generic_tag = 'P_VAL'") == []
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string="tags.generic_tag != 'P_VAL'"))
        # test search returns appropriate run (same key different values per run)
        assert self._search(experiment_id, filter_string="tags.generic_2 = 'some value'") == [r1]
        assert self._search(experiment_id, filter_string="tags.generic_2 = 'another value'") == [r2]
        assert self._search(experiment_id, filter_string="tags.generic_tag = 'wrong_val'") == []
        assert self._search(experiment_id, filter_string="tags.generic_tag != 'p_val'") == []
        assert sorted(
            [r1, r2],
        ) == sorted(
            self._search(experiment_id, filter_string="tags.generic_tag != 'wrong_val'"),
        )
        assert sorted(
            [r1, r2],
        ) == sorted(
            self._search(experiment_id, filter_string="tags.generic_2 != 'wrong_val'"),
        )
        assert self._search(experiment_id, filter_string="tags.p_a = 'abc'") == [r1]
        assert self._search(experiment_id, filter_string="tags.p_b = 'ABC'") == [r2]
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE '%other%'") == [r2]
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE '%Other%'") == []
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE 'other%'") == []
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE '%other'") == []
        assert self._search(experiment_id, filter_string="tags.generic_2 LIKE 'other'") == []
        assert self._search(experiment_id, filter_string="tags.generic_2 ILIKE '%Other%'") == [r2]
        assert self._search(
            experiment_id,
            filter_string="tags.generic_2 ILIKE '%Other%' and tags.generic_tag = 'p_val'",
        ) == [r2]
        assert self._search(
            experiment_id,
            filter_string="tags.generic_2 ILIKE '%Other%' and " "tags.generic_tag ILIKE 'p_val'",
        ) == [r2]

    def test_search_metrics(self):
        experiment_id = self._experiment_factory("search_metric")
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.log_metric(r1, entities.Metric("common", 1.0, 1, 0))
        self.store.log_metric(r2, entities.Metric("common", 1.0, 1, 0))

        self.store.log_metric(r1, entities.Metric("measure_a", 1.0, 1, 0))
        self.store.log_metric(r2, entities.Metric("measure_a", 200.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("measure_a", 400.0, 3, 0))

        self.store.log_metric(r1, entities.Metric("m_a", 2.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 3.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 4.0, 8, 0))  # this is last timestamp
        self.store.log_metric(r2, entities.Metric("m_b", 8.0, 3, 0))

        filter_string = "metrics.common = 1.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common > 0.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common >= 0.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common < 4.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common <= 4.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.common != 1.0"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "metrics.common >= 3.0"
        assert self._search(experiment_id, filter_string) == []

        filter_string = "metrics.common <= 0.75"
        assert self._search(experiment_id, filter_string) == []

        # tests for same metric name across runs with different values and timestamps
        filter_string = "metrics.measure_a > 0.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a < 50.0"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "metrics.measure_a < 1000.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a != -12.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        filter_string = "metrics.measure_a > 50.0"
        assert self._search(experiment_id, filter_string) == [r2]

        filter_string = "metrics.measure_a = 1.0"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "metrics.measure_a = 400.0"
        assert self._search(experiment_id, filter_string) == [r2]

        # test search with unique metric keys
        filter_string = "metrics.m_a > 1.0"
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = "metrics.m_b > 1.0"
        assert self._search(experiment_id, filter_string) == [r2]

        # there is a recorded metric this threshold but not last timestamp
        filter_string = "metrics.m_b > 5.0"
        assert self._search(experiment_id, filter_string) == []

        # metrics matches last reported timestamp for 'm_b'
        filter_string = "metrics.m_b = 4.0"
        assert self._search(experiment_id, filter_string) == [r2]

    def test_search_attrs(self):
        e1 = self._experiment_factory("search_attributes_1")
        r1 = self._run_factory(self._get_run_configs(experiment_id=e1)).info.run_id

        e2 = self._experiment_factory("search_attrs_2")
        r2 = self._run_factory(self._get_run_configs(experiment_id=e2)).info.run_id

        filter_string = ""
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = "attribute.status != 'blah'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = "attribute.status = '{}'".format(RunStatus.to_string(RunStatus.RUNNING))
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        # change status for one of the runs
        self.store.update_run_info(r2, RunStatus.FAILED, 300, None)

        filter_string = "attribute.status = 'RUNNING'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = "attribute.status = 'FAILED'"
        assert self._search([e1, e2], filter_string) == [r2]

        filter_string = "attribute.status != 'SCHEDULED'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = "attribute.status = 'SCHEDULED'"
        assert self._search([e1, e2], filter_string) == []

        filter_string = "attribute.status = 'KILLED'"
        assert self._search([e1, e2], filter_string) == []

        if is_windows():
            expected_artifact_uri = (
                pathlib.Path.cwd().joinpath(ARTIFACT_URI, e1, r1, "artifacts").as_uri()
            )
            filter_string = f"attr.artifact_uri = '{expected_artifact_uri}'"
        else:
            expected_artifact_uri = (
                pathlib.Path.cwd().joinpath(ARTIFACT_URI, e1, r1, "artifacts").as_posix()
            )
            filter_string = f"attr.artifact_uri = '{expected_artifact_uri}'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = "attr.artifact_uri = '{}/{}/{}/artifacts'".format(
            ARTIFACT_URI, e1.upper(), r1.upper()
        )
        assert self._search([e1, e2], filter_string) == []

        filter_string = "attr.artifact_uri != '{}/{}/{}/artifacts'".format(
            ARTIFACT_URI, e1.upper(), r1.upper()
        )
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = f"attr.artifact_uri = '{ARTIFACT_URI}/{e2}/{r1}/artifacts'"
        assert self._search([e1, e2], filter_string) == []

        filter_string = "attribute.artifact_uri = 'random_artifact_path'"
        assert self._search([e1, e2], filter_string) == []

        filter_string = "attribute.artifact_uri != 'random_artifact_path'"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search([e1, e2], filter_string))

        filter_string = f"attribute.artifact_uri LIKE '%{r1}%'"
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = "attribute.artifact_uri LIKE '%{}%'".format(r1[:16])
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = "attribute.artifact_uri LIKE '%{}%'".format(r1[-16:])
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = "attribute.artifact_uri LIKE '%{}%'".format(r1.upper())
        assert self._search([e1, e2], filter_string) == []

        filter_string = "attribute.artifact_uri ILIKE '%{}%'".format(r1.upper())
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = "attribute.artifact_uri ILIKE '%{}%'".format(r1[:16].upper())
        assert self._search([e1, e2], filter_string) == [r1]

        filter_string = "attribute.artifact_uri ILIKE '%{}%'".format(r1[-16:].upper())
        assert self._search([e1, e2], filter_string) == [r1]

        for k, v in {"experiment_id": e1, "lifecycle_stage": "ACTIVE"}.items():
            with pytest.raises(MlflowException, match=r"Invalid attribute key '.+' specified"):
                self._search([e1, e2], f"attribute.{k} = '{v}'")

    def test_search_full(self):
        experiment_id = self._experiment_factory("search_params")
        r1 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        r2 = self._run_factory(self._get_run_configs(experiment_id)).info.run_id

        self.store.log_param(r1, entities.Param("generic_param", "p_val"))
        self.store.log_param(r2, entities.Param("generic_param", "p_val"))

        self.store.log_param(r1, entities.Param("p_a", "abc"))
        self.store.log_param(r2, entities.Param("p_b", "ABC"))

        self.store.log_metric(r1, entities.Metric("common", 1.0, 1, 0))
        self.store.log_metric(r2, entities.Metric("common", 1.0, 1, 0))

        self.store.log_metric(r1, entities.Metric("m_a", 2.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 3.0, 2, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 4.0, 8, 0))
        self.store.log_metric(r2, entities.Metric("m_b", 8.0, 3, 0))

        filter_string = "params.generic_param = 'p_val' and metrics.common = 1.0"
        assert sorted(
            [r1, r2],
        ) == sorted(self._search(experiment_id, filter_string))

        # all params and metrics match
        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 and metrics.m_a > 1.0"
        )
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 "
            "and metrics.m_a > 1.0 and params.p_a LIKE 'a%'"
        )
        assert self._search(experiment_id, filter_string) == [r1]

        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 "
            "and metrics.m_a > 1.0 and params.p_a LIKE 'A%'"
        )
        assert self._search(experiment_id, filter_string) == []

        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 "
            "and metrics.m_a > 1.0 and params.p_a ILIKE 'A%'"
        )
        assert self._search(experiment_id, filter_string) == [r1]

        # test with mismatch param
        filter_string = (
            "params.random_bad_name = 'p_val' and metrics.common = 1.0 and metrics.m_a > 1.0"
        )
        assert self._search(experiment_id, filter_string) == []

        # test with mismatch metric
        filter_string = (
            "params.generic_param = 'p_val' and metrics.common = 1.0 and metrics.m_a > 100.0"
        )
        assert self._search(experiment_id, filter_string) == []

    def test_search_with_max_results(self):
        exp = self._experiment_factory("search_with_max_results")
        runs = [
            self._run_factory(self._get_run_configs(exp, start_time=r)).info.run_id
            for r in range(1200)
        ]
        # reverse the ordering, since we created in increasing order of start_time
        runs.reverse()

        assert runs[:1000] == self._search(exp)
        for n in [0, 1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 1200, 2000]:
            assert runs[: min(1200, n)] == self._search(exp, max_results=n)

        with pytest.raises(
                MlflowException, match=r"Invalid value for request parameter max_results"
        ):
            self._search(exp, max_results=int(1e10))

    def test_search_with_deterministic_max_results(self):
        exp = self._experiment_factory("test_search_with_deterministic_max_results")
        # Create 10 runs with the same start_time.
        # Sort based on run_id
        runs = sorted(
            [
                self._run_factory(self._get_run_configs(exp, start_time=10)).info.run_id
                for r in range(10)
            ]
        )
        for n in [0, 1, 2, 4, 8, 10, 20]:
            assert runs[: min(10, n)] == self._search(exp, max_results=n)

    
    def test_search_runs_pagination(self):
        exp = self._experiment_factory("test_search_runs_pagination")
        # test returned token behavior
        runs = sorted(
            [
                self._run_factory(self._get_run_configs(exp, start_time=10)).info.run_id
                for r in range(10)
            ]
        )
        result = self.store.search_runs([exp], None, ViewType.ALL, max_results=4)
        assert [r.info.run_id for r in result] == runs[0:4]
        assert result.token is not None
        result = self.store.search_runs(
            [exp], None, ViewType.ALL, max_results=4, page_token=result.token
        )
        assert [r.info.run_id for r in result] == runs[4:8]
        assert result.token is not None
        result = self.store.search_runs(
            [exp], None, ViewType.ALL, max_results=4, page_token=result.token
        )
        assert [r.info.run_id for r in result] == runs[8:]
        assert result.token is None

    def test_search_runs_run_name(self):
        exp_id = self._experiment_factory("test_search_runs_pagination")
        run1 = self._run_factory(dict(self._get_run_configs(exp_id), run_name="run_name1"))
        run2 = self._run_factory(dict(self._get_run_configs(exp_id), run_name="run_name2"))
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'run_name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run1.info.run_id]
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.`Run name` = 'run_name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run1.info.run_id]
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.`run name` = 'run_name2'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run2.info.run_id]
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.`Run Name` = 'run_name2'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run2.info.run_id]
        result = self.store.search_runs(
            [exp_id],
            filter_string="tags.`mlflow.runName` = 'run_name2'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run2.info.run_id]

        self.store.update_run_info(
            run1.info.run_id,
            RunStatus.FINISHED,
            end_time=run1.info.end_time,
            run_name="new_run_name1",
        )
        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'new_run_name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run1.info.run_id]

        result = self.store.search_runs(
            [exp_id],
            filter_string="tags.`mlflow.runName` = 'new_run_name1'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run1.info.run_id]

    def test_search_runs_run_id(self):
        exp_id = self._experiment_factory("test_search_runs_run_id")
        # Set start_time to ensure the search result is deterministic
        run1 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=1))
        run2 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=2))
        run_id1 = run1.info.run_id
        run_id2 = run2.info.run_id

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id = '{run_id1}'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id != '{run_id1}'",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run_id2]

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id IN ('{run_id1}')",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id NOT IN ('{run_id1}')",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run_id2]

        for filter_string in [
            f"attributes.run_id IN ('{run_id1}','{run_id2}')",
            f"attributes.run_id IN ('{run_id1}', '{run_id2}')",
            f"attributes.run_id IN ('{run_id1}',  '{run_id2}')",
        ]:
            result = self.store.search_runs(
                [exp_id], filter_string=filter_string, run_view_type=ViewType.ACTIVE_ONLY
            )
            assert [r.info.run_id for r in result] == [run_id2, run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string=f"attributes.run_id NOT IN ('{run_id1}', '{run_id2}')",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert result == []

    
    def test_search_runs_start_time_alias(self):
        exp_id = self._experiment_factory("test_search_runs_start_time_alias")
        # Set start_time to ensure the search result is deterministic
        run1 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=1))
        run2 = self._run_factory(dict(self._get_run_configs(exp_id), start_time=2))
        run_id1 = run1.info.run_id
        run_id2 = run2.info.run_id

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'name'",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.start_time DESC"],
        )
        assert [r.info.run_id for r in result] == [run_id2, run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'name'",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.created ASC"],
        )
        assert [r.info.run_id for r in result] == [run_id1, run_id2]

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.run_name = 'name'",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.Created DESC"],
        )
        assert [r.info.run_id for r in result] == [run_id2, run_id1]

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.start_time > 0",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert {r.info.run_id for r in result} == {run_id1, run_id2}

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.created > 1",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert [r.info.run_id for r in result] == [run_id2]

        result = self.store.search_runs(
            [exp_id],
            filter_string="attributes.Created > 2",
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        assert result == []

    def test_log_batch(self):
        experiment_id = self._experiment_factory("log_batch")
        run_id = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        metric_entities = [Metric("m1", 0.87, 12345, 0), Metric("m2", 0.49, 12345, 1)]
        param_entities = [Param("p1", "p1val"), Param("p2", "p2val")]
        tag_entities = [
            RunTag("t1", "t1val"),
            RunTag("t2", "t2val"),
            RunTag(MLFLOW_RUN_NAME, "my_run"),
        ]
        self.store.log_batch(
            run_id=run_id, metrics=metric_entities, params=param_entities, tags=tag_entities
        )
        run = self.store.get_run(run_id)
        assert run.data.tags == {"t1": "t1val", "t2": "t2val", MLFLOW_RUN_NAME: "my_run"}
        assert run.data.params == {"p1": "p1val", "p2": "p2val"}
        metric_histories = sum(
            [self.store.get_metric_history(run_id, key) for key in run.data.metrics], []
        )
        metrics = [(m.key, m.value, m.timestamp, m.step) for m in metric_histories]
        assert set(metrics) == {("m1", 0.87, 12345, 0), ("m2", 0.49, 12345, 1)}

    def test_log_batch_limits(self):
        # Test that log batch at the maximum allowed request size succeeds (i.e doesn't hit
        # SQL limitations, etc)
        experiment_id = self._experiment_factory("log_batch_limits")
        run_id = self._run_factory(self._get_run_configs(experiment_id)).info.run_id
        metric_tuples = [("m%s" % i, i, 12345, i * 2) for i in range(1000)]
        metric_entities = [Metric(*metric_tuple) for metric_tuple in metric_tuples]
        self.store.log_batch(run_id=run_id, metrics=metric_entities, params=[], tags=[])
        run = self.store.get_run(run_id)
        metric_histories = sum(
            [self.store.get_metric_history(run_id, key) for key in run.data.metrics], []
        )
        metrics = [(m.key, m.value, m.timestamp, m.step) for m in metric_histories]
        assert set(metrics) == set(metric_tuples)

    def test_log_batch_param_overwrite_disallowed(self):
        # Test that attempting to overwrite a param via log_batch results in an exception and that
        # no partial data is logged
        run = self._run_factory()
        tkey = "my-param"
        param = entities.Param(tkey, "orig-val")
        self.store.log_param(run.info.run_id, param)

        overwrite_param = entities.Param(tkey, "newval")
        tag = entities.RunTag("tag-key", "tag-val")
        metric = entities.Metric("metric-key", 3.0, 12345, 0)
        with pytest.raises(
                MlflowException, match=r"Changing param values is not allowed"
        ) as exception_context:
            self.store.log_batch(
                run.info.run_id, metrics=[metric], params=[overwrite_param], tags=[tag]
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self._verify_logged(self.store, run.info.run_id, metrics=[metric], params=[param], tags=[])

    def test_log_batch_with_unchanged_and_new_params(self):
        """
        Test case to ensure the following code works:
        ---------------------------------------------
        mlflow.log_params({"a": 0, "b": 1})
        mlflow.log_params({"a": 0, "c": 2})
        ---------------------------------------------
        """
        run = self._run_factory()
        self.store.log_batch(
            run.info.run_id,
            metrics=[],
            params=[entities.Param("a", "0"), entities.Param("b", "1")],
            tags=[],
        )
        self.store.log_batch(
            run.info.run_id,
            metrics=[],
            params=[entities.Param("a", "0"), entities.Param("c", "2")],
            tags=[],
        )
        self._verify_logged(
            self.store,
            run.info.run_id,
            metrics=[],
            params=[entities.Param("a", "0"), entities.Param("b", "1"), entities.Param("c", "2")],
            tags=[],
        )

    def test_log_batch_param_overwrite_disallowed_single_req(self):
        # Test that attempting to overwrite a param via log_batch results in an exception
        run = self._run_factory()
        pkey = "common-key"
        param0 = entities.Param(pkey, "orig-val")
        param1 = entities.Param(pkey, "newval")
        tag = entities.RunTag("tag-key", "tag-val")
        metric = entities.Metric("metric-key", 3.0, 12345, 0)
        with pytest.raises(
                MlflowException, match=r"^Duplicate parameter keys have been submitted.*"
        ) as exception_context:
            self.store.log_batch(
                run.info.run_id, metrics=[metric], params=[param0, param1], tags=[tag]
            )
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=[], tags=[])

    def test_log_batch_accepts_empty_payload(self):
        run = self._run_factory()
        self.store.log_batch(run.info.run_id, metrics=[], params=[], tags=[])
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=[], tags=[])

    def test_log_batch_internal_error(self):
        # Verify that internal errors during the DB save step for log_batch result in
        # MlflowExceptions
        run = self._run_factory()

        def _raise_exception_fn(*args, **kwargs):  # pylint: disable=unused-argument
            raise Exception("Some internal error")

        package = "mlflow_mongostore.mongo_store.MongoStore"
        with mock.patch(package + "._log_metrics") as metric_mock, mock.patch(
                package + "._log_param"
        ) as param_mock, mock.patch(package + "._set_tag") as tags_mock:
            metric_mock.side_effect = _raise_exception_fn
            param_mock.side_effect = _raise_exception_fn
            tags_mock.side_effect = _raise_exception_fn
            for kwargs in [
                {"metrics": [Metric("a", 3, 1, 0)]},
                {"params": [Param("b", "c")]},
                {"tags": [RunTag("c", "d")]},
            ]:
                log_batch_kwargs = {"metrics": [], "params": [], "tags": []}
                log_batch_kwargs.update(kwargs)
                with pytest.raises(MlflowException, match=r"Some internal error"):
                    self.store.log_batch(run.info.run_id, **log_batch_kwargs)

    def test_log_batch_nonexistent_run(self):
        nonexistent_run_id = uuid.uuid4().hex
        with pytest.raises(
                MlflowException, match=rf"Run with id={nonexistent_run_id} not found"
        ) as exception_context:
            self.store.log_batch(nonexistent_run_id, [], [], [])
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    def test_log_batch_params_idempotency(self):
        run = self._run_factory()
        params = [Param("p-key", "p-val")]
        self.store.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
        self.store.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=params, tags=[])

    def test_log_batch_tags_idempotency(self):
        run = self._run_factory()
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")]
        )
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")]
        )
        self._verify_logged(
            self.store, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")]
        )

    def test_log_batch_allows_tag_overwrite(self):
        run = self._run_factory()
        self.store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "val")])
        self.store.log_batch(
            run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")]
        )
        self._verify_logged(
            self.store, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")]
        )

    def test_log_batch_allows_tag_overwrite_single_req(self):
        run = self._run_factory()
        tags = [RunTag("t-key", "val"), RunTag("t-key", "newval")]
        self.store.log_batch(run.info.run_id, metrics=[], params=[], tags=tags)
        self._verify_logged(self.store, run.info.run_id, metrics=[], params=[], tags=[tags[-1]])

    def test_log_batch_metrics(self):
        run = self._run_factory()
        mongo_run = self.store._get_run(run.info.run_id)

        tkey = "blahmetric"
        tval = 100.0
        metric = entities.Metric(tkey, tval, get_current_time_millis(), 0)
        metric2 = entities.Metric(tkey, tval, get_current_time_millis() + 2, 0)
        nan_metric = entities.Metric("NaN", float("nan"), 0, 0)
        pos_inf_metric = entities.Metric("PosInf", float("inf"), 0, 0)
        neg_inf_metric = entities.Metric("NegInf", -float("inf"), 0, 0)

        # duplicate metric and metric2 values should be eliminated
        metrics = [metric, metric2, nan_metric, pos_inf_metric, neg_inf_metric, metric, metric2]
        self.store._log_metrics(mongo_run, metrics)

        run = self.store.get_run(run.info.run_id)
        assert tkey in run.data.metrics and run.data.metrics[tkey] == tval

        # SQL store _get_run method returns full history of recorded metrics.
        # Should return duplicates as well
        # MLflow RunData contains only the last reported values for metrics.
        sql_run_metrics = self.store._get_run(run.info.run_id).metrics
        assert len(sql_run_metrics) == 5
        assert len(run.data.metrics) == 4
        assert math.isnan(run.data.metrics["NaN"])
        assert run.data.metrics["PosInf"] == 1.7976931348623157e308
        assert run.data.metrics["NegInf"] == -1.7976931348623157e308

    def test_log_batch_same_metric_repeated_single_req(self):
        run = self._run_factory()
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
        self._verify_logged(
            self.store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[]
        )

    def test_log_batch_same_metric_repeated_multiple_reqs(self):
        run = self._run_factory()
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0], tags=[])
        self._verify_logged(self.store, run.info.run_id, params=[], metrics=[metric0], tags=[])
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric1], tags=[])
        self._verify_logged(
            self.store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[]
        )

    def test_log_batch_same_metrics_repeated_multiple_reqs(self):
        run = self._run_factory()
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
        self._verify_logged(
            self.store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[]
        )
        self.store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
        self._verify_logged(
            self.store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[]
        )

    def test_log_batch_null_metrics(self):
        run = self._run_factory()

        tkey = "blahmetric"
        tval = None
        metric_1 = entities.Metric(tkey, tval, get_current_time_millis(), 0)

        tkey = "blahmetric2"
        tval = None
        metric_2 = entities.Metric(tkey, tval, get_current_time_millis(), 0)

        metrics = [metric_1, metric_2]

        with pytest.raises(
                MlflowException, match=r"Got invalid value None for metric"
        ) as exception_context:
            self.store.log_batch(run.info.run_id, metrics=metrics, params=[], tags=[])
        assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_log_batch_params_max_length_value(self):
        run = self._run_factory()
        param_entities = [Param("long param", "x" * 500), Param("short param", "xyz")]
        expected_param_entities = [Param("long param", "x" * 500), Param("short param", "xyz")]
        self.store.log_batch(run.info.run_id, [], param_entities, [])
        self._verify_logged(self.store, run.info.run_id, [], expected_param_entities, [])
        param_entities = [Param("long param", "x" * 1000)]
        with pytest.raises(MlflowException, match="exceeded length"):
            self.store.log_batch(run.info.run_id, [], param_entities, [])

    def _generate_large_data(self, nb_runs=1000):
        experiment_id = self.store.create_experiment("test_experiment")

        current_run = 0

        run_ids = []

        for _ in range(nb_runs):
            run_id = self.store.create_run(
                experiment_id=experiment_id,
                start_time=current_run,
                tags=[],
                user_id="Anderson",
                run_name="name",
            ).info.run_uuid

            run_ids.append(run_id)
            mongo_run = self.store._get_run(run_id)

            for i in range(100):
                metric = {
                    "key": "mkey_%s" % i,
                    "value": i,
                    "timestamp": i * 2,
                    "step": i * 3,
                    "is_nan": False,
                    "run_uuid": run_id,
                }
                MongoMetric(**metric).save()
                tag = {
                    "key": "tkey_%s" % i,
                    "value": "tval_%s" % (current_run % 10),
                }
                mongo_run.update(push__tags=MongoTag(**tag))
                param = {
                    "key": "pkey_%s" % i,
                    "value": "pval_%s" % ((current_run + 1) % 11),
                }
                mongo_run.update(push__params=MongoParam(**param))

            mongo_run.update(push__latest_metrics=MongoLatestMetric(**{
                    "key": "mkey_0",
                    "value": current_run,
                    "timestamp": 100 * 2,
                    "step": 100 * 3,
                    "is_nan": False,
                }))

            current_run += 1

        return experiment_id, run_ids

    def test_search_runs_returns_expected_results_with_large_experiment(self):
        """
        This case tests the SQLAlchemyStore implementation of the SearchRuns API to ensure
        that search queries over an experiment containing many runs, each with a large number
        of metrics, parameters, and tags, are performant and return the expected results.
        """
        experiment_id, run_ids = self._generate_large_data(30)

        run_results = self.store.search_runs([experiment_id], None, ViewType.ALL, max_results=10)
        assert len(run_results) == 10
        # runs are sorted by desc start_time
        assert [run.info.run_id for run in run_results] == list(reversed(run_ids[20:]))

    # TODO
    # def test_search_runs_correctly_filters_large_data(self):
    #     experiment_id, _ = self._generate_large_data(30)
    #
    #     run_results = self.store.search_runs(
    #         [experiment_id],
    #         "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 ",
    #         ViewType.ALL,
    #         max_results=50,
    #     )
    #     assert len(run_results) == 20
    #
    #     run_results = self.store.search_runs(
    #         [experiment_id],
    #         "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 and tags.tkey_0 = 'tval_0' ",
    #         ViewType.ALL,
    #         max_results=10,
    #     )
    #     assert len(run_results) == 2  # 20 runs between 9 and 26, 2 of which have a 0 tkey_0 value
    #
    #     run_results = self.store.search_runs(
    #         [experiment_id],
    #         "metrics.mkey_0 < 26 and metrics.mkey_0 > 5 "
    #         "and tags.tkey_0 = 'tval_0' "
    #         "and params.pkey_0 = 'pval_0'",
    #         ViewType.ALL,
    #         max_results=5,
    #     )
    #     assert len(run_results) == 1  # 2 runs on previous request, 1 of which has a 0 pkey_0 value

    def test_search_runs_keep_all_runs_when_sorting(self):
        experiment_id = self.store.create_experiment("test_experiment1")

        r1 = self.store.create_run(
            experiment_id=experiment_id, start_time=0, tags=[], user_id="Me", run_name="name"
        ).info.run_uuid
        r2 = self.store.create_run(
            experiment_id=experiment_id, start_time=0, tags=[], user_id="Me", run_name="name"
        ).info.run_uuid
        self.store.set_tag(r1, RunTag(key="t1", value="1"))
        self.store.set_tag(r1, RunTag(key="t2", value="1"))
        self.store.set_tag(r2, RunTag(key="t2", value="1"))

        run_results = self.store.search_runs(
            [experiment_id], None, ViewType.ALL, max_results=1000, order_by=["tag.t1"]
        )
        assert len(run_results) == 2

    def test_try_get_run_tag(self):
        run = self._run_factory()
        self.store.set_tag(run.info.run_id, entities.RunTag("k1", "v1"))
        self.store.set_tag(run.info.run_id, entities.RunTag("k2", "v2"))

        mongo_run = self.store._get_run(run.info.run_id)

        tags = mongo_run.get_tags_by_key("k0")
        assert len(tags) == 0

        tags = mongo_run.get_tags_by_key("k1")
        assert len(tags) == 1
        tag = tags[0]
        assert tag.key == "k1"
        assert tag.value == "v1"

        tags = mongo_run.get_tags_by_key("k2")
        assert len(tags) == 1
        tag = tags[0]
        assert tag.key == "k2"
        assert tag.value == "v2"

    def test_get_metric_history_on_non_existent_metric_key(self):
        experiment_id = self._experiment_factory("test_exp")[0]
        run = self.store.create_run(
            experiment_id=experiment_id, user_id="user", start_time=0, tags=[], run_name="name"
        )
        run_id = run.info.run_id
        metrics = self.store.get_metric_history(run_id, "test_metric")
        assert metrics == []

    def test_record_logged_model(self):
        store = self.get_store()
        run_id = self.create_test_run().info.run_id
        m = Model(artifact_path="model/path", run_id=run_id, flavors={"tf": "flavor body"})
        store.record_logged_model(run_id, m)
        self._verify_logged(
            store,
            run_id=run_id,
            params=[],
            metrics=[],
            tags=[RunTag(MLFLOW_LOGGED_MODELS, json.dumps([m.to_dict()]))],
        )
        m2 = Model(
            artifact_path="some/other/path", run_id=run_id, flavors={"R": {"property": "value"}}
        )
        store.record_logged_model(run_id, m2)
        self._verify_logged(
            store,
            run_id,
            params=[],
            metrics=[],
            tags=[RunTag(MLFLOW_LOGGED_MODELS, json.dumps([m.to_dict(), m2.to_dict()]))],
        )
        m3 = Model(
            artifact_path="some/other/path2", run_id=run_id, flavors={"R2": {"property": "value"}}
        )
        store.record_logged_model(run_id, m3)
        self._verify_logged(
            store,
            run_id,
            params=[],
            metrics=[],
            tags=[
                RunTag(MLFLOW_LOGGED_MODELS, json.dumps([m.to_dict(), m2.to_dict(), m3.to_dict()]))
            ],
        )
        with pytest.raises(
                TypeError,
                match="Argument 'mlflow_model' should be mlflow.models.Model, got '<class 'dict'>'",
        ):
            store.record_logged_model(run_id, m.to_dict())


def test_get_attribute_name():
    assert MongoRun.get_attribute_name("artifact_uri") == "artifact_uri"
    assert MongoRun.get_attribute_name("status") == "status"
    assert MongoRun.get_attribute_name("start_time") == "start_time"
    assert MongoRun.get_attribute_name("end_time") == "end_time"
    assert MongoRun.get_attribute_name("deleted_time") == "deleted_time"
    assert MongoRun.get_attribute_name("run_name") == "name"
    assert MongoRun.get_attribute_name("run_id") == "run_uuid"

    # we want this to break if a searchable or orderable attribute has been added
    # and not referred to in this test
    # searchable attributes are also orderable
    assert len(entities.RunInfo.get_orderable_attributes()) == 7
