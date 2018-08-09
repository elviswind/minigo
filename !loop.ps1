While ($true)
{
	python .\rl_loop.py selfplay --base_dir='.'
	python .\rl_loop.py train estimator_working_dir --base_dir='.' --model_dir='estimator_working_dir'
}