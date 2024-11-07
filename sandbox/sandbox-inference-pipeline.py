from utils.pipelines.inference  import InferencePipeline

pipe = InferencePipeline("./runs/run-0009 (wiki-2500)", device="cuda:0")

res = pipe.generate("Hi dear model", max_tokens=200)

print(res)