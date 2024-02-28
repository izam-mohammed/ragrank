import logging
import os
from pathlib import Path

print(f"current working dir - {os.getcwd()}")
input_val = input("press ENTER to proceed, 'q' to exit, 'up' to go root: ")
input_val = input_val.lower()
while input_val:
    if input_val == "up":
        os.chdir("../")
    elif not input_val:
        break
    else:
        exit()


logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = "Ragrank"
docs_folder = "docs/docs"
metrics = [
    "response_related/response_relevancy",
    "safity_related/thretening",
    "efficiency_related/response_time",
    "novelty_related/creativity",
    "coverage_related/domain_coverage",
    "interpretebility_related/transparancy",
    "robustness_related/Adversarial_robustness",
    "chat_related/satisfaction",
    "language_related/grammar",
    "coding_related/code_quality",
]

list_of_files = [
    f"src/{project_name}/llm/__init__.py",
    f"src/{project_name}/llm/base.py",
    f"src/{project_name}/llm/openai.py",
    f"src/{project_name}/llm/langchain.py",
    f"src/{project_name}/embedding/__init__.py",
    f"src/{project_name}/embedding/base.py",
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
    f"src/{project_name}/_version.py",
    f"src/{project_name}/_api.py",
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
    f"src/{project_name}/agents/eval_agent.py",
    f"src/{project_name}/validation/__init__.py",
    f"src/{project_name}/validation/base.py",
    f"src/{project_name}/validation/datase_validator.py",
    f"src/{project_name}/validation/metric_validator.py",
    f"src/{project_name}/validation/llm_validator.py",
    f"src/{project_name}/validation/embedding_validator.py",
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
    "docs/api_reference/create_api_rst.py",
    # docs related
    f"{docs_folder}/index.md",
    f"{docs_folder}/getting_started/index.md",
    f"{docs_folder}/getting_started/quick_start.md",
    f"{docs_folder}/getting_started/basic_evluation.md",
    f"{docs_folder}/getting_started/test_data_generation.md",
    f"{docs_folder}/core_concepts/index.md",
    f"{docs_folder}/core_concepts/what_is_llm.md",
    f"{docs_folder}/core_concepts/what_and_why_rag.md",
    f"{docs_folder}/core_concepts/why_evaluation.md",
    f"{docs_folder}/core_concepts/why_ragrank.md",
    f"{docs_folder}/metrics/index.md",
    f"{docs_folder}/evaluation/index.md",
    f"{docs_folder}/evaluation/comprehensive_evaluation.md",
    f"{docs_folder}/evaluation/with_custom_data.md",
    f"{docs_folder}/evaluation/with_custom_llm_and_embedding.md",
    f"{docs_folder}/evaluation/eval_chain.md",
    f"{docs_folder}/evaluation/with_decorators.md",
    f"{docs_folder}/evaluation/with_custom_prompts.md",
    f"{docs_folder}/evaluation/callbacks_and_cache.md",
    f"{docs_folder}/integrations/framework/index.md",
    f"{docs_folder}/integrations/llm/index.md",
    f"{docs_folder}/integrations/evalution/index.md",
    f"{docs_folder}/integrations/vectordb/index.md",
    f"{docs_folder}/metrics/index.md",
    f"{docs_folder}/metrics/response_related/index.md",
    f"{docs_folder}/metrics/safity_related/index.md",
    f"{docs_folder}/metrics/efficiency_related/index.md",
    f"{docs_folder}/metrics/novelty_related/index.md",
    f"{docs_folder}/metrics/coverage_related/index.md",
    f"{docs_folder}/metrics/interpretebility_related/index.md",
    f"{docs_folder}/metrics/robustness_related/index.md",
    f"{docs_folder}/metrics/chat_related/index.md",
    f"{docs_folder}/metrics/language_related/index.md",
    f"{docs_folder}/metrics/coding_related/index.md",
    f"{docs_folder}/api_reference/index.md",
    f"{docs_folder}/community/index.md",
    f"{docs_folder}/more/index.md",
    f"{docs_folder}/more/about_us.md",
    f"{docs_folder}/more/contributing.md",
    f"{docs_folder}/more/security.md",
]

# adding the metric
list_of_files += [f"{docs_folder}/metrics/{metric}.md" for metric in metrics]

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
