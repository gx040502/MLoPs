# Architecture Diagram

A high level diagram that describe the abstract function of each component.
The diagram is create according to guidance in [this medium][medium-software-diagrams-guide] post.
Diagram levels are split into:
1. System Context diagrams
2. Container diagrams
3. Component diagrams
4. [Code diagrams][uml-class-diagram]

[medium-software-diagrams-guide]: https://medium.com/@jancalve/writing-good-software-architecture-diagrams-15c51eca4ce7 "Writing good sofware architecture diagrams"
[uml-class-diagram]: https://www.visual-paradigm.com/guide/uml-unified-modeling-language/uml-class-diagram-tutorial/ "UML Class Diagram"

![System Context Diagram](/docs/assests/system_context.svg)

## Contents

*   [Getting started](#getting-started)
    *   [Environment Configuration](#environment-configuration)
    *   [Usage](#usage)
*   [License](#license)
*   [Sources](#sources)

## Getting Started

### Environment Configuration

Use conda to create the virutal environment.
```bash
# /home/user/path/to/repo/
conda env create --prefix /opt/miniconda3/envs/agentic --file requirements.yaml

# List down available conda environments
conda env list
```

### Usage
```bash
# Activate the environment
conda activate agentic

# Run the program
python3 main.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Sources

[Markdown Cheat Sheet][markdown-cheatsheet] - Get a fast overview of the syntax

[//]: # "Source definitions"
[react-markdown]: https://github.com/remarkjs/react-markdown "React-markdown project"
[blog-post-templates]: https://backlinko.com/hub/content/blog-post-templates "Backlinko blog post templates"
[about-markdown]: https://www.markdownguide.org/getting-started/ "Introduction to markdown"
[markdown-cheatsheet]: https://www.markdownguide.org/cheat-sheet/ "Markdown Cheat Sheet"


