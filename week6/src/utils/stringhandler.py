import re

def contains_number(string_list):
    pattern = re.compile(r'\d')
    for string in string_list:
        if pattern.search(string):
            return True
    return False
