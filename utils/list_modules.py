from modulefinder import ModuleFinder
finder = ModuleFinder()
finder.run_script("../workflow/runReconstruction.py")
# Get names of all the imported modules
names = list(finder.modules.keys())
# Get a sorted list of the root modules imported
basemods = sorted(set([name.split('.')[0] for name in names]))
# Print it nicely
print("\n".join(basemods))