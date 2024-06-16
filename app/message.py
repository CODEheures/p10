import logging

class Message():

    def __init__(self, mode='notebook'):
        self.mode = mode
    
    def print(self, message):
        print(message) if self.mode == 'notebook' else logging.info(message)
