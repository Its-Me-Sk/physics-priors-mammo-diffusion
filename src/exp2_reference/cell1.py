# ===================== CELL 1 — setup (Experiment 2) (run in EVERY new session; needs Internet ON) =====================
import os, re, subprocess, time
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
os.chdir("/kaggle/working")
if not os.path.isdir("/kaggle/working/U24_synthesis"):
    subprocess.run("git clone -q https://github.com/Its-Me-Sk/U24_synthesis.git", shell=True, check=True)
if not os.path.exists("/tmp/pip_done"):
    subprocess.run("cd /kaggle/working/U24_synthesis && grep -v xformers requirements.txt > /tmp/req.txt && "
                   "pip install -q -r /tmp/req.txt && "
                   "pip install -q numpy==1.26.4 pandas==2.2.2 --force-reinstall && "
                   "pip install -q --no-deps torch-fidelity", shell=True, check=True)
    open("/tmp/pip_done", "w").write("ok")
p = "/kaggle/working/U24_synthesis/lightning_synthesis/model_architectures/monai_ddpm.py"
s = open(p).read()
s2 = re.sub(r"use_flash_attention\s*=\s*True", "use_flash_attention=False", s)   # xformers not installed
if s2 != s:
    open(p, "w").write(s2)
# models.py (oracle loader) has a hard-coded path from the original author's machine -> point it to the repo's config.yaml
m = "/kaggle/working/U24_synthesis/models.py"
s = open(m).read()
s2 = re.sub(r"open\(\s*['\"]/home/[^'\"]*config\.yaml['\"]", "open('/kaggle/working/U24_synthesis/config.yaml'", s)
if s2 != s:
    open(m, "w").write(s2)
print("models.py config path ->", "patched" if s2 != s else "already ok")
print("Cell 1 done. Kernel restart NOT needed (all GPU work runs in subprocesses).")
