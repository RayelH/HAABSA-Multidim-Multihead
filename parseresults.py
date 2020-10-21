from config import *

file_path = "cross_results_" + "2016" + "/multidim_new" + "2016" + '.txt'
write_path = "cross_results_" + "2016" + "/multidim_new_short" + "2016" + '.txt'



with open(file_path, 'r') as handle, open(write_path, 'w') as filewrite:
    for line in handle:
        if line.startswith('Config') or line.startswith('Max') or line.startswith('Min'):
             print(line)
             filewrite.write(line)
             # Write to file, etc.