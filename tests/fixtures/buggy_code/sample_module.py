"""
Sample buggy Python code for testing the refactoring workflow.
This module contains various code quality issues that should be detected and fixed.
"""

import os,sys
import json

def calculate_area(length,width):
    # Missing docstring
    # Poor variable naming
    # No type hints
    result=length*width
    return result

class DataProcessor:
    def __init__(self):
        self.data=[]
        
    def process_data(self,input_data):
        # Long line that exceeds recommended length and has multiple issues including poor formatting and missing error handling
        processed_data = [item.strip().upper() for item in input_data if item is not None and len(item) > 0 and item.strip() != ""]
        return processed_data
    
    def save_to_file(self, filename):
        # No error handling
        # Hardcoded file operations
        with open(filename, 'w') as f:
            json.dump(self.data, f)

def unused_function():
    # This function is never called
    x = 1
    y = 2
    return x + y

# Global variable (bad practice)
GLOBAL_COUNTER = 0

def increment_counter():
    global GLOBAL_COUNTER
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER

# Missing main guard
if True:
    print("This should be in a main guard")
    processor = DataProcessor()
    area = calculate_area(5, 10)
    print(f"Area: {area}")