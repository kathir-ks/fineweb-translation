import os
import subprocess


subsets = ['CC-MAIN-2013-20', 'CC-MAIN-2013-48']

output = subprocess.check_output(['gcloud',
                                  'alpha', 
                                  'compute', 
                                  'tpus', 
                                  'tpu-vm', 
                                  'ssh', 
                                  'main-1',
                                  '--zone=us-central2-b', 
                                  '--tunnel-through-iap', 
                                  '--command=\'cd fineweb-translation; python3 inference.py --name HuggingFaceFW/fineweb-edu --subset CC-MAIN-2013-20 --batch_size 1024 --bucket gs://indic-llama-data \''])

print(output)