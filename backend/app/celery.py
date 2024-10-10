from celery import Celery
from celery.schedules import crontab
from app.core.config import settings

app = Celery(
    settings.PROJECT_NAME,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

app.conf.broker_connection_retry_on_startup = True


app.autodiscover_tasks(['app'])

