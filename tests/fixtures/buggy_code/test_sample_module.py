"""
Test file for sample_module.py
Contains basic tests that should pass after refactoring.
"""

import unittest
from sample_module import calculate_area, DataProcessor, increment_counter


class TestSampleModule(unittest.TestCase):
    
    def test_calculate_area(self):
        """Test area calculation function."""
        self.assertEqual(calculate_area(5, 10), 50)
        self.assertEqual(calculate_area(0, 10), 0)
        self.assertEqual(calculate_area(3, 4), 12)
    
    def test_data_processor_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        self.assertEqual(processor.data, [])
    
    def test_process_data(self):
        """Test data processing functionality."""
        processor = DataProcessor()
        input_data = ["  hello  ", "WORLD", "", None, "test"]
        result = processor.process_data(input_data)
        expected = ["HELLO", "WORLD", "TEST"]
        self.assertEqual(result, expected)
    
    def test_increment_counter(self):
        """Test counter increment function."""
        # Reset global counter for test
        import sample_module
        sample_module.GLOBAL_COUNTER = 0
        
        result1 = increment_counter()
        result2 = increment_counter()
        
        self.assertEqual(result1, 1)
        self.assertEqual(result2, 2)


if __name__ == '__main__':
    unittest.main()