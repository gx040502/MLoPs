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

> [!note]
> Ensure that  your system is running **python>=3.12** before proceeding

```bash
# Create python virtual environment
python3 -m venv .venv

# Acticate python virtual environment
source .venv/bin/activate
```

### Usage

>>> [!note]
Ensure you have activated the previously created virtual environment before
proceeding. **requirements.txt** will be installed by pip when installing the
project for development.
>>> 

```bash
# Extract the obtained zip file to current project root directory
tar -xzvf archive.tar.gz

# Install the project for development mode
python3 -m pip3 install -e . -v

# Remove the unused opencv depending on desktop or server
python3 -m pip3 uninstall opencv-python # if running on server
python3 -m pip3 uninstall opencv-python-headless # if running on desktop that has GUI

# Run the VLM Annotation Dashboard
python3 -m VLMAnnotate
```

>>> [!tip]
Try `pip install --force-reinstall opencv-python-headless` or 
`pip install --force-reinstall opencv-python` if there is error 
"**cv2 is not found**"
>>>

## License
[Closed Source License](LICENSE.txt)
