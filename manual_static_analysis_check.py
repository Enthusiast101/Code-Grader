#using this as a temp file to check for syntax errors

import json

from grading_agent.analyzers.static_analysis import run_static_analysis
from grading_agent.sandbox.docker_sandbox import DockerSandbox


source = """#include <stdio.h>

int main(void) {
    int unused = 10;
    printf("Hello\\n");
    return 0;
}
"""

with DockerSandbox(memory_limit="64m") as sandbox:
    result = run_static_analysis(
        sandbox=sandbox,
        source_code=source,
        warning_penalty=0.1,
    )

print(json.dumps(result, indent=2))
