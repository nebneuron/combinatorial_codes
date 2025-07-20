import sys
from combinatorial_codes import utils

print("Testing C extension status", file=sys.stderr)
print(f"utils.tf = {utils.tf}", file=sys.stderr)

if utils.tf is not None:
    print("SUCCESS: C extensions are working!", file=sys.stderr)
else:
    print("PROBLEM: C extensions not detected", file=sys.stderr)

# Now test the status function
from combinatorial_codes import check_c_extension_status
print("Running status check...", file=sys.stderr)
result = check_c_extension_status()
print(f"Status result: {result}", file=sys.stderr)
