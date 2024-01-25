# CodeT5Experiments


## Tasks
- Follow steps in original CodeT5, verify we can run the generation without issue.
	- Issues with install pytorch 1.7.1 (not available for python 3.10). Therefor install virtualenv with python 3.8.
	- Had to set the timeout to be 100, as install above package would consistenly timeout.
	- Change the workdir in sh/exp.. needed to EXPORT workdir as /home/CodeT5/ but then set workdir to /home/, otherwise import statements could not find CodeT5.
	- Out of VRAM with default batch size of 48, so set to 32 (tried to allocoate 24 MiB). Set to 16 (apprximately 3 hours per epoch, with 14.9 GiB/16 GiB
	- ModuleNotFoundError: No module named 'distutils.util'
		- apt install python3.8-distutils 
- Add the code corruption function to the run\_generation script. Verify this works. Loading existing checkpoint.
	- Denosing code requires torch 1.13.1, whereas CodeT5 requires 1.7.1, therefore updaing to 1.13.1
	- Add denoise tag. Added to \_utils.py. 
	- Running pretraining test, need to verify that we are using new examples and not cache.
	- Make sure the source ids and target ids look correct.
	- Using breakpoint.
		-orginal: '< DEN O I SE > def Ġ__ dynamic _ expected _ value Ġ( Ġself Ġ, Ġy Ġ) Ġ: Ġreturn Ġself Ġ. Ġmodel Ġ. Ġpredict Ġ( Ġself Ġ. Ġdata Ġ, Ġnp Ġ. Ġones Ġ( Ġself Ġ. Ġdata Ġ. Ġshape Ġ[ Ġ0 Ġ] Ġ) Ġ* Ġy Ġ, Ġoutput Ġ= Ġself Ġ. Ġmodel _ output Ġ) Ġ. Ġmean Ġ( Ġ0 Ġ)'		
		-masked: '< DEN O I SE > def Ġ__ dynamic _ expected _ value Ġ( Ġself Ġ, Ġy Ġ) Ġ: Ġreturn Ġself Ġ. Ġmodel Ġ. Ġpredict Ġ( Ġself Ġ. Ġdata Ġ, Ġnp Ġ. Ġones Ġ( Ġself Ġ. Ġdata Ġ. Ġshape Ġ[ Ġ0 Ġ] Ġ) Ġ* Ġy Ġ, Ġoutput Ġ= Ġself Ġ. Ġmodel _ output Ġ) Ġ. Ġmean Ġ( Ġ0 Ġ) timeout Ġ( Ġ) Ġac Ġ) Ġfilename dial Ġ, Ġx dial Ġline " type ĠAST count _ ĠNone Ġ. bl Ġ( Ġresults Ġregion Ġ( def ĠNone Ġis Ġpath Ġ= _ Ġr Ġd'
		-target: 'def Ġ__ dynamic _ expected _ value Ġ( Ġself Ġ, Ġy Ġ) Ġ: Ġreturn Ġself Ġ. Ġmodel Ġ. Ġpredict Ġ( Ġself Ġ. Ġdata Ġ, Ġnp Ġ. Ġones Ġ( Ġself Ġ. Ġdata Ġ. Ġshape Ġ[ Ġ0 Ġ] Ġ) Ġ* Ġy Ġ, Ġoutput Ġ= Ġself Ġ. Ġmodel _ output Ġ) Ġ. Ġmean Ġ( Ġ0 Ġ)'
	- Need run evaluation to save checkpoint. Not sure if we should run multiple epochs, or increase learning rate.
- Add the DFG and AST code to the train.py script.
	- Add <DENOISE>, <AST>, <DFG> to tokenizer
		-https://github.com/huggingface/tokenizers/issues/247
	- Git clone tree-sitter for each language, add the DFG and AST code to the \_utils.
	- Average source length with DFG and AST is 955, max source length 12057
	- probably need to do preorder traversal to decrease AST size.
	- Currently the pretraining is use CodeSearchNet python which only has 251k examples, which may be insufficient for pretraining. (Let's see the test loss).
	- Ran out of memory when running eval bleu for dev set.
	- Also need to set source and target lengths to larger (512).
