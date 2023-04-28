#!/usr/bin/env python

# REF [site] >>
#	http://docs.celeryproject.org/en/latest/getting-started/first-steps-with-celery.html
#	http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html
def celery_tasks_test():
	from celery_tasks import add

	if False:
		# This is a handy shortcut to the apply_async() method that gives greater control of the task execution.
		result = add.delay(6, 7)

		# Results are not enabled by default.
		# In order to do remote procedure calls or keep track of task results in a database, you will need to configure Celery to use a result backend.
		#print(f"{result.get()=}.")
	else:
		# If you want to keep track of the tasks' states, Celery needs to store or send the states somewhere.
		# There are several built-in result backends to choose from:
		#	SQLAlchemy/Django ORM.
		#	MongoDB.
		#	Memcached.
		#	Redis.
		#	RPC (RabbitMQ/AMQP).

		# This is a handy shortcut to the apply_async() method that gives greater control of the task execution.
		result = add.delay(6, 7)

		print(f"{result.ready()=}.")  # Returns whether the task has finished processing or not.
		print(f"{result.get(timeout=1)=}.")
		# In case the task raised an exception, get() will re-raise the exception, but you can override this by specifying the propagate argument.
		print(f"{result.get(propagate=False)=}.")
		# If the task raised an exception, you can also gain access to the original traceback.
		print(f"{result.traceback=}.")

def main():
	celery_tasks_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
