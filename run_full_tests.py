import subprocess, sys, pathlib
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
    capture_output=True, text=True,
    cwd=pathlib.Path(__file__).parent,
)
out = result.stdout + result.stderr
pathlib.Path("tests/full_run.txt").write_text(out, encoding="utf-8")
print(out)
