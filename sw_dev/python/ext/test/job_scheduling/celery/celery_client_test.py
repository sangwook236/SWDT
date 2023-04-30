#!/usr/bin/env python

# REF [site] >>
#	http://docs.celeryproject.org/en/latest/getting-started/first-steps-with-celery.html
#	http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html
def celery_tasks_test():
	from celery_tasks import add, sub, mul, div

	# Results are not enabled by default.
	# In order to do remote procedure calls or keep track of task results in a database, you will need to configure Celery to use a result backend.

	# If you want to keep track of the tasks' states, Celery needs to store or send the states somewhere.
	# There are several built-in result backends to choose from:
	#	SQLAlchemy/Django ORM.
	#	MongoDB.
	#	Memcached.
	#	Redis.
	#	RPC (RabbitMQ/AMQP).

	# This is a handy shortcut to the apply_async() method that gives greater control of the task execution.
	result = add.delay(6, 7)
	#result = sub.delay(13, 7)
	#result = mul.delay(2, 3)
	#result = div.delay(10, 5)
	#result = div.delay(6, 0)

	# REF [site] >> https://docs.celeryq.dev/en/stable/reference/celery.result.html
	print(f"{type(result)=}.")  # celery.result.AsyncResult.

	print(f"{result.ready()=}.")  # Returns whether the task has finished processing or not.
	while not result.ready():
		pass
	print(f"{result.ready()=}.")

	print(f"{result.successful()=}.")
	print(f"{result.failed()=}.")

	print(f"{result.task_id=}.")
	print(f"{result.name=}.")
	print(f"{result.app =}.")
	print(f"{result.backend =}.")
	print(f"{result.date_done=}.")
	print(f"{result.result=}.")
	print(f"{result.state=}.")
	print(f"{result.status=}.")
	print(f"{result.info=}.")
	print(f"{result.retries=}.")
	print(f"{result.ignored=}.")
	print(f"{result.queue=}.")
	print(f"{result.children=}.")
	print(f"{result.worker=}.")

	try:
		print(f"{result.get()=}.")
		print(f"{result.get(timeout=1)=}.")
	except Exception as ex:
		print(f"Exception raised: {ex}.")
	# In case the task raised an exception, get() will re-raise the exception, but you can override this by specifying the propagate argument.
	print(f"{result.get(propagate=False)=}.")
	# If the task raised an exception, you can also gain access to the original traceback.
	print("Traceback:")
	print(result.traceback)

	result.forget()
	print(f"{result.ready()=}.")
	print(f"{result.get()=}.")

def main():
	celery_tasks_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
