import sys, os, warnings, traceback
sys.path.append(os.path.dirname(__file__))
print(os.path.dirname(__file__))

try:
    from ensembler import potentials
    from ensembler import integrator
    from ensembler import system
    from ensembler import ensemble
    from ensembler import conditions
except Exception as err:
    print()
    traceback.print_exc()
    #exit(1)
try:
    from ensembler import visualisation
except Exception as err:
    warnings.warn("Could not find matplotlib or ipywidgets therefore no Visualisation.")

