# s10_pool.py - concatenate the seed-42 and seed-43 image sets of one arm (same noise bank -> image i stays paired)
import os, sys, numpy as np
OUT = os.environ["PA_OUT"]; out, a, b = sys.argv[1:4]
A, B = np.load(os.path.join(OUT, f"gen_{a}.npz")), np.load(os.path.join(OUT, f"gen_{b}.npz"))
assert (A["y"] == B["y"]).all(), "label order differs"
np.savez_compressed(os.path.join(OUT, f"gen_{out}.npz"), x=np.concatenate([A["x"], B["x"]]), y=np.concatenate([A["y"], B["y"]]))
print(f"[S10-pool] gen_{out}.npz = {a} + {b} ({len(A['y']) + len(B['y'])} images)", flush=True)
