from typing import TYPE_CHECKING

class TaskOutput:

    def __init__(
        self, 
        task_id = None, 
        output = None,
        success = None,
        exception = None,
    ):
        self.id = task_id
        self.output = output
        self.success = success
        self.exception = exception