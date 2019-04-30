"""
List all modules that a script import
"""
from modulefinder import ModuleFinder
finder = ModuleFinder()
finder.run_script("tests/testReconAllData.py")
# Get names of all the imported modules
names = list(finder.modules.keys())
# Get a sorted list of the root modules imported
basemods = sorted(set([name.split('.')[0] for name in names]))
# Print it nicely
print("\n".join(basemods))