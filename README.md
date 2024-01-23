# CodeT5Experiments


## Tasks
- Follow steps in original CodeT5, verify we can run the generation without issue.
	- Issues with install pytorch 1.7.1 (not available for python 3.10). Therefor install virtualenv with python 3.8.
	- Had to set the timeout to be 100, as install above package would consistenly timeout.
	- Change the workdir in sh/exp.. needed to EXPORT workdir as /home/CodeT5/ but then set workdir to /home/, otherwise import statements could not find CodeT5.
	- Out of VRAM with default batch size of 48, so set to 32 (tried to allocoate 24 MiB). Set to 16 (apprximately 3 hours per epoch, with 14.9 GiB/16 GiB
	- ModuleNotFoundError: No module named 'distutils.util'
		- apt install python3.8-distutils 
- Add the code corruption function to the run_generation script. Verify this works. Loading existing checkpoint.
	- Add denoise tag.
	- Use Python codesearch net code. Potentially other code too.
- Add the DFG and AST code to the train.py script.
