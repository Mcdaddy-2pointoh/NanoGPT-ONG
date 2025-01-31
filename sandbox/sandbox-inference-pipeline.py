from pipelines.inference.inference  import InferencePipeline

pipe = InferencePipeline("./runs/run-0035", device="cuda:0")
prompt = """"""
res = pipe.generate(prompt, max_tokens=150)

print(prompt)
print()
print(res)