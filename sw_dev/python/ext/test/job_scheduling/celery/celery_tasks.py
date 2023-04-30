#!/usr/bin/env python

# REF [site] >>
#	http://docs.celeryproject.org/en/latest/getting-started/first-steps-with-celery.html
#	http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html

from celery import Celery

#app = Celery("tasks", broker="pyamqp://guest@localhost//")  # With no result backend.
#app = Celery("tasks", broker="redis://localhost:6379/0")  # With no result backend.
#app = Celery("tasks", backend="rpc://", broker="pyamqp://guest@localhost//")
app = Celery("tasks", backend="redis://localhost:6379/0", broker="pyamqp://guest@localhost//")  # Very commonly used.
#app = Celery("tasks", backend="redis://localhost:6379/0", broker="redis://localhost:6379/0")

"""
app.conf.task_serializer = "json"

app.conf.update(
	task_serializer="json",
	accept_content=["json"],  # Ignore other content.
	result_serializer="json",
	timezone="Europe/Oslo",
	enable_utc=True,
)

In celeryconfig.py:
	broker_url = "pyamqp://guest@localhost//"
	#broker_url = "redis://localhost:6379/0"
	result_backend = "rpc://"
	#result_backend = "redis://localhost:6379/0"

	task_serializer = "json"
	result_serializer = "json"
	accept_content = ["json"]
	timezone = "Europe/Oslo"
	enable_utc = True

	# You'd route a misbehaving task to a dedicated queue.
	task_routes = {
		"celery_tasks.add": "low-priority",
	}

	# You could rate limit the task instead, so that only 10 tasks of this type can be processed in a minute (10/m).
	task_annotations = {
		"celery_tasks.add": {"rate_limit": "10/m"}
	}

# A module named celeryconfig.py must be available to load from the current directory or on the Python path.
app.config_from_object("celeryconfig")
"""

@app.task
def add(x, y):
	return x + y

@app.task
def sub(x, y):
	return x - y

@app.task
def mul(x, y):
	return x * y

@app.task
def div(x, y):
	return x / y

def main():
	app.conf.broker_url = "pyamqp://guest@localhost//"
	#app.conf.broker_url = "redis://localhost:6379/0"

#--------------------------------------------------------------------

# Usage:
#	Running the Celery worker server:
#		celery -A celery_tasks worker --loglevel=INFO
#
#	If you're using RabbitMQ or Redis as the broker then you can also direct the workers to set a new rate limit for the task at runtime:
#		celery -A celery_tasks control rate_limit tasks.add 10/m

if "__main__" == __name__:
	main()
