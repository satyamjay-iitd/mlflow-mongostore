import datetime
import numbers

from mlflow.entities import Experiment, ExperimentTag, RunTag, RunInfo, RunData, Run, Metric, Param, SourceType, \
    RunStatus
from mlflow.entities import LifecycleStage
from mlflow.utils.mlflow_tags import _get_run_name_from_tags
from mlflow.utils.time_utils import get_current_time_millis

from mongoengine import Document, StringField, ListField, EmbeddedDocument, \
    EmbeddedDocumentField, FloatField, IntField, BooleanField, LongField, ReferenceField, \
    CASCADE, EmbeddedDocumentListField


def compare_attr(val1, comp, val2):
    if type(val1) != type(val2):
        return False

    is_numeric = isinstance(val1, numbers.Number)
    if is_numeric:
        if comp == ">":
            return val1 > val2
        elif comp == ">=":
            return val1 > val2
        elif comp == "!=":
            return val1 > val2
        elif comp == "=":
            return val1 > val2
        elif comp == "<":
            return val1 > val2
        elif comp == "<=":
            return val1 > val2
        return False
    else:
        if comp == "=":
            return val1 == val2
        elif comp == "!=":
            return val1 == val2
        elif comp == "LIKE":
            return val1.contains(val2)
        elif comp == "ILIKE":
            return val1.lower().contains(val2.lower())


class SequenceId(Document):
    collection_name = StringField(primary_key=True)
    sequence_value = LongField()


class MongoExperimentTag(EmbeddedDocument):
    key = StringField(required=True)
    value = StringField(required=True)

    def to_mlflow_entity(self) -> ExperimentTag:
        return ExperimentTag(key=self.key,
                             value=self.value)


class MongoExperiment(Document):
    exp_id = StringField(primary_key=True)
    name = StringField(required=True, max_length=200)
    artifact_location = StringField(max_length=256)
    lifecycle_stage = StringField(max_length=50, default=LifecycleStage.ACTIVE)
    tags = ListField(EmbeddedDocumentField(MongoExperimentTag))
    creation_time = LongField()
    last_update_time = LongField()

    meta = {'collection': 'mlflow_experiments'}

    def to_mlflow_entity(self) -> Experiment:
        return Experiment(
            experiment_id=str(self.id),
            name=self.name,
            artifact_location=self.artifact_location,
            lifecycle_stage=self.lifecycle_stage,
            tags=[t.to_mlflow_entity() for t in self.tags],
            creation_time=self.creation_time,
            last_update_time=self.last_update_time
        )


class MongoTag(EmbeddedDocument):
    key = StringField(required=True)
    value = StringField(required=True)

    def to_mlflow_entity(self) -> RunTag:
        return RunTag(
            key=self.key,
            value=self.value)


class MongoParam(EmbeddedDocument):
    key = StringField(required=True)
    value = StringField(required=True)

    def to_mlflow_entity(self) -> Param:
        return Param(
            key=self.key,
            value=self.value)


class MongoMetric(Document):
    key = StringField(required=True)
    timestamp = LongField(default=get_current_time_millis)
    step = LongField(required=True, default=0)
    is_nan = BooleanField(required=True, default=False)
    run_uuid = StringField(required=True)
    value = FloatField(unique_with=["key", "timestamp", "step", "run_uuid"],
                       required=True)

    meta = {'collection': "mlflow_metric"}

    def to_mlflow_entity(self) -> Metric:
        return Metric(
            key=self.key,
            value=self.value if not self.is_nan else float("nan"),
            timestamp=self.timestamp,
            step=self.step)


class MongoLatestMetric(EmbeddedDocument):
    key = StringField()
    value = FloatField()
    timestamp = LongField()
    step = IntField()
    is_nan = BooleanField()

    def to_mlflow_entity(self) -> Metric:
        return Metric(
            key=self.key,
            value=self.value if not self.is_nan else float("nan"),
            timestamp=self.timestamp,
            step=self.step)


class MongoRun(Document):
    run_uuid = StringField(primary_key=True, required=True, max_length=32)
    name = StringField(max_length=250)
    source_type = StringField(max_length=20, default=SourceType.to_string(SourceType.LOCAL))
    source_name = StringField(max_length=500)
    entry_point_name = StringField(max_length=50)
    user_id = StringField(max_length=256, default="")
    status = StringField(max_length=20, default=RunStatus.to_string(RunStatus.SCHEDULED))
    start_time = LongField(default=get_current_time_millis)
    end_time = LongField()
    deleted_time = LongField()
    source_version = StringField(max_length=50)
    lifecycle_stage = StringField(max_length=20, default=LifecycleStage.ACTIVE)
    artifact_uri = StringField(max_length=200)

    experiment_id = ReferenceField('MongoExperiment', reverse_delete_rule=CASCADE)

    latest_metrics = ListField(EmbeddedDocumentField(MongoLatestMetric))
    params = ListField(EmbeddedDocumentField(MongoParam))
    # tags = ListField(EmbeddedDocumentField(MongoTag))
    tags = EmbeddedDocumentListField(MongoTag)

    meta = {'collection': "mlflow_runs"}

    def to_mlflow_entity(self) -> Run:
        run_info = RunInfo(
            run_uuid=self.run_uuid,
            run_id=self.run_uuid,
            run_name=self.name,
            experiment_id=str(self.experiment_id.id),
            user_id=self.user_id,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            lifecycle_stage=self.lifecycle_stage,
            artifact_uri=self.artifact_uri,
        )

        tags = [t.to_mlflow_entity() for t in self.tags]
        run_data = RunData(
            metrics=[m.to_mlflow_entity() for m in self.latest_metrics],
            params=[p.to_mlflow_entity() for p in self.params],
            tags=tags)

        if not run_info.run_name:
            run_name = _get_run_name_from_tags(tags)
            if run_name:
                run_info._set_run_name(run_name)

        return Run(run_info=run_info, run_data=run_data)

    @staticmethod
    def get_attribute_name(mlflow_attribute_name):
        return {"run_name": "name", "run_id": "run_uuid"}.get(
            mlflow_attribute_name, mlflow_attribute_name
        )

    @property
    def metrics(self):
        return MongoMetric.objects(run_uuid=self.run_uuid)

    def get_param_by_key(self, key):
        params = list(filter(lambda param: param.key == key, self.params))
        return params[0] if params else None

    def get_tags_by_key(self, key):
        return list(filter(lambda param: param.key == key, self.tags))


if __name__ == "__main__":
    from mongoengine import connect

    ex = MongoExperiment.objects()

    mex = MongoExperiment(name="tst2", artifact_location="xyz.com",
                          tags=[MongoExperimentTag(key="k1", value="v1"),
                                MongoExperimentTag(key="k2", value="v2")])

    mex.save()
    # print(mex.id)

    tags_dict = {"k1": "v1", "k2": "v2"}
    run_tags = [MongoTag(key=key, value=value) for key, value in tags_dict.items()]
    run = MongoRun(
        run_id="1",
        experiment_id="1", user_id="4",
        status="Running",
        start_time=datetime.datetime.now(), end_time=None,
        lifecycle_stage=LifecycleStage.ACTIVE, artifact_uri="artifact_location",
        tags=run_tags)

    run.save()

    run.update(push__tags=MongoTag(key="k3", value="v3"))
