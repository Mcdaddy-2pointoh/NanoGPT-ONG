from pipelines.inference  import InferencePipeline

pipe = InferencePipeline("./runs/run-0016/checkpoints", device="cuda:0")
prompt = "How many states in Australia? "
res = pipe.generate(prompt, max_tokens=150)

print(prompt, res)