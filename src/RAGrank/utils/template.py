import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = "RAGrank"

list_of_files = [
    f"src/{project_name}/llm/__init__.py",
    f"src/{project_name}/llm/base.py",
    f"src/{project_name}/llm/openai.py",
    f"src/{project_name}/llm/langchain.py",
    f"src/{project_name}/embedding/__init__.py",
    f"src/{project_name}/embedchain/base.py",
    f"src/{project_name}/embedding/langchain.py",
    f"src/{project_name}/metrics/__init__.py",
    f"src/{project_name}/metrics/base.py",
    f"src/{project_name}/metrics/context_relevancy.py",
    f"src/{project_name}/evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/integration/__init__.py",
    f"src/{project_name}/prompt/__init__.py",
    f"src/{project_name}/prompt/base.py",
    f"src/{project_name}/prompt/templates.py",
    f"src/{project_name}/dataset/__init__.py",
    f"src/{project_name}/dataset/base.py",
    f"src/{project_name}/dataset/generator.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/base.py",
    f"src/{project_name}/config/run_config.py",
    f"src/{project_name}/_version.py"
    f"src/{project_name}/api.py",
    f"src/{project_name}/constants.py",
    f"src/{project_name}/cache.py",
    f"src/{project_name}/runner.py",
    f"src/{project_name}/exceptions.py",
    f"src/{project_name}/callbacks.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/bridge/__init__.py",
    f"src/{project_name}/bridge/pydante.py",
    f"src/{project_name}/bridge/langchain.py",
    f"src/{project_name}/decorators.py",
    f"src/{project_name}/agents/__init__.py",
    f"src/{project_name}/agents/base.py",
    f"src/{project_name}/agents/ragrankbaseagent.py",
    f"src/{project_name}/validation.py",
    f"src/{project_name}/hub/__init__.py",
    ".github/ISSUE_TEMPLATE/bug-report.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/documentation.yml",
    ".github/DISCUSSION_TEMPLATE/ideas.yml",
    ".github/DISCUSSION_TEMPLATE/q-a.yml",
    ".github/CONTRIBUTING.md",
    ".github/CODE_OF_CONDUCT.md",
    ".github/FUNDING.yaml",
    ".github/workflows/python-cli.yml",
    ".github/workflows/greetings.yml",
    "SECURITY.md",
    ".gitattributes",
    "setup.py",
    ".devcontainer/README.md",
    ".devcontainer/devcontainer.json",
    ".devcontainer/docker-compose.yaml",
    ".readthedocs.yaml",
    
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory; {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")