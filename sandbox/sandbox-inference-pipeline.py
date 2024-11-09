from utils.pipelines.inference  import InferencePipeline

pipe = InferencePipeline("./runs/run-0010", device="cuda:0")
prompt = "What is India?"
res = pipe.generate(prompt, max_tokens=200)

print(prompt, res)