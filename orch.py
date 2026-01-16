import subprocess

commands = [
    ["python", "opt_sagsl_alpha.py", "--dataset", "citeseer"],
    ["python", "opt_sagsl_beta.py", "--dataset", "citeseer"],
    ["python", "opt_sagsl_delta.py", "--dataset", "citeseer"],
    ["python", "opt_sagsl_gamma.py", "--dataset", "citeseer"],
    ["python", "opt_sagsl_lambda.py", "--dataset", "citeseer"],
]

processes = []

print("Launching processes...")


for cmd in commands:
    p = subprocess.Popen(cmd)
    processes.append(p)

print("All processes launched! Waiting for them to finish...")

for p in processes:
    p.wait()

print("All Done.")