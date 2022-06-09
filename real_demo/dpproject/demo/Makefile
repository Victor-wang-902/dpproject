check:
	squeue -u zw2374

attach:
	srun --jobid $(id) --pty /bin/bash

clean:
	rm -f *.err *.out

run:
	sbatch demo_greene.slurm

cancel:
	scancel  $(id)
