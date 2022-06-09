check:
	squeue -u zw2374

attach:
	srun --jobid $(id) --pty /bin/bash

clean:
	rm -f *.err *.out

eval:
	sbatch demo_greene_eval.slurm

train:
	sbatch demo.slurm

test:
	sbatch demo_greene_test.slurm

cancel:
	scancel  $(id)
