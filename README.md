# Architecture Diagram

A high level diagram that describe the abstract function of each component.
The diagram is created according to guidance in [this medium][medium-software-diagrams-guide] post that uses [C4 model](c4-model) for visualizing software architecture.
Diagram levels are split into:
1. System Context diagrams
2. Container diagrams
3. Component diagrams
4. [Code diagrams][uml-class-diagram]

[medium-software-diagrams-guide]: https://medium.com/@jancalve/writing-good-software-architecture-diagrams-15c51eca4ce7 "Writing good sofware architecture diagrams"
[uml-class-diagram]: https://www.visual-paradigm.com/guide/uml-unified-modeling-language/uml-class-diagram-tutorial/ "UML Class Diagram"
[c4-model]: https://c4model.com/ "C4 Model"

> [!note]
> The Architecture Diagrams are still in developement, 8-10-2025 (dd-mm-yyy)

<details>

<summary>System Context Diagram</summary>

![System Context Diagram](/docs/assests/system_context.svg "System Context Diagram")

</details>

## Contents

*   [Getting started](#getting-started)
    *   [Environment Configuration](#environment-configuration)
    *   [Usage](#usage)
*   [License](#license)

## Getting Started

### Environment Configuration

Use conda to create the virutal environment.
```bash
# Create the conda environment using the `YAML` file
conda env create --prefix /opt/miniconda3/envs/agentic --file requirements.yaml

# List down available conda environments
conda env list
```

### Usage

```bash
# Activate the environment
conda activate agentic

# Extract the obtained zip file to current project root directory
tar -xzvf archive.tar.gz

# Install the project for development mode
python3 -m pip3 install -e . -v

# Run the VLM Annotation Dashboard
python3 -m VLMAnnotate
```

## License
[Closed Source License](LICENSE.txt)
