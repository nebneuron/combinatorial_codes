#!/usr/bin/env python3

import sys
print("Testing C extension status...")

try:
    from combinatorial_codes import utils
    print(f"utils.tf = {utils.tf}")
    print(f"type(utils.tf) = {type(utils.tf)}")
    
    if utils.tf is not None:
        print("C extension is available!")
        print(f"Available functions: {[x for x in dir(utils.tf) if not x.startswith('_')]}")
    else:
        print("C extension is NOT available (utils.tf is None)")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
