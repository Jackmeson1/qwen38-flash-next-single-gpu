"""Write a safetensors index for the MTP draft folder that lists only the mtp.* tensors of Intel's extra file."""
import json
import sys

d = sys.argv[1]
idx = json.load(open(f"{d}/intel/model.safetensors.index.json"))
wm = {k: "model_extra_tensors.safetensors" for k in idx["weight_map"] if k.startswith("mtp.")}
json.dump({"metadata": {}, "weight_map": wm}, open(f"{d}/model.safetensors.index.json", "w"))
print(len(wm), "MTP tensors")  # 1565
