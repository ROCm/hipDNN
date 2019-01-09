import os
import sys

abs_path = os.path.abspath('.')
path = abs_path.rsplit('/',1)[0]
py_path = os.path.join(path,'test','utils','csv')
sys.path.append(py_path)

import compare_results