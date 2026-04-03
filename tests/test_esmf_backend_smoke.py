import os
import subprocess


def test_esmfold_backend_smoke():
    output_dir = "outputs/test_trace_ci"

    cmd = [
        "python",
        "run_pretrained_esmf.py",
        "--fasta",
        "examples/monomer/fasta_dir_6KWC/6KWC.fasta",
        "--out",
        output_dir,
        "--trace_mode",
        "attention+activations",
        "--device",
        "cpu",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    # expected outputs
    assert os.path.exists(f"{output_dir}/meta.json")
    assert os.path.exists(f"{output_dir}/structure/predicted.pdb")
    assert os.path.exists(f"{output_dir}/trace")

    # check tensor count
    trace_files = []
    for root, _, files in os.walk(f"{output_dir}/trace"):
        trace_files += [f for f in files if f.endswith(".pt")]

    assert len(trace_files) >= 36

