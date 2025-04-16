import sys
from src.logger import logging  # Assuming you want to use standard logging; change if you have a custom logger

def error_message_detail(error, error_detail: sys):
    _, _, exe_tb = error_detail.exc_info()
    file_name = exe_tb.tb_frame.f_code.co_filename if exe_tb else "Unknown file"
    line_number = exe_tb.tb_lineno if exe_tb else "Unknown line"
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error))
    return error_message

class CustomException(Exception):  # Fixed typo from CustomeException
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
