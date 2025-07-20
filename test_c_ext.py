from combinatorial_codes import utils

print('utils.tf:', utils.tf)
print('Type:', type(utils.tf))

if utils.tf:
    print('Functions:', [x for x in dir(utils.tf) if not x.startswith('_')])
    print('C extensions are working!')
else:
    print('C extensions NOT working')
