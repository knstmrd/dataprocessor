class DataProcessorError(Exception):
    def __init__(self, msg):
        super(DataProcessorError, self).__init__(msg)
