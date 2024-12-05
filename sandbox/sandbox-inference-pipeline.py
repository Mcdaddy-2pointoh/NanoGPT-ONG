from pipelines.inference.inference  import InferencePipeline

pipe = InferencePipeline("./runs/run-0017", device="cuda:0")
prompt = """There are other relevant issues in this whole debate, among which are serious concerns at the alarmingly slow pace of the destruction of weaponized chemical agents by the major possessor States."""
res = pipe.generate(prompt, max_tokens=150)

print(prompt)
print()
print(res)