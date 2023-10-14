#DAGs

## bash operator dag
```
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner":"data-science-wizard",
    "retries":2,
    "retry_delay":timedelta(minutes=2)
}

with DAG(

    dag_id = "our_first_dag_v4",
    description = "this is our first airflow dag",
    default_args =default_args,
    start_date = datetime(2023,10,10, 1,30),
    schedule_interval = "@daily"

) as dag:
    task1 = BashOperator(
        task_id = "first_task",
        bash_command = "echo hello world, this is the first task"
    )
    task2 = BashOperator(
        task_id = "second_task",
        bash_command = "echo this is the second task, i will run after first."
    )
    task3 = BashOperator(
        task_id = "third_task",
        bash_command = "echo this is the third task, i will run after first along with second."
    )
    
    #task1.set_downstream(task2)
    #task1.set_downstream(task3)
    task1 >> task2
    task1 >> task3
```
