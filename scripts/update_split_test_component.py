# !/usr/bin/env python3
import subprocess
import sys

flows_dir = "../flows"
flow_file = flows_dir + "/ingestion_flow.json"
code_file = flows_dir + "/components/split_text.py"
display_name = "Split Text"


def main():
    read_component()
    # write_component()

def read_component():
    metadata_module = None # "mypkg.flow_meta"  # OPTIONAL
    match_index = None  # OPTIONAL
    output = code_file # OPTIONAL

    # Build the command
    cmd = [sys.executable, "extract_flow_component.py", "--flow-file", flow_file]

    if display_name:
        cmd += ["--display-name", display_name]
    if metadata_module:
        cmd += ["--metadata-module", metadata_module]
    if match_index is not None:
        cmd += ["--match-index", str(match_index)]
    if output:
        cmd += ["--output", output]

    # Run the command
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)

def write_component():
    # Build the command
    cmd = [sys.executable, "update_flow_components.py",
        "--code-file", code_file,
        "--display-name", display_name,
        "--flows-dir", flows_dir
    ]

    # Run the command
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)


if __name__ == "__main__":
    main()