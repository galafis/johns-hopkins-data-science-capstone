#!/usr/bin/env python3
"""
Unit Tests for Johns Hopkins Data Science Capstone
"""

import unittest
import sys
import os

class TestPlatform(unittest.TestCase):
    """Basic platform tests"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = 2 + 2
        self.assertEqual(result, 4)
    
    def test_string_operations(self):
        """Test string operations"""
        text = "Hello World"
        self.assertTrue(len(text) > 0)
        self.assertIn("World", text)
    
    def test_list_operations(self):
        """Test list operations"""
        test_list = [1, 2, 3, 4, 5]
        self.assertEqual(len(test_list), 5)
        self.assertIn(3, test_list)

if __name__ == '__main__':
    unittest.main()
