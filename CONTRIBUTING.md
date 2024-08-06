# Checklist

We are glad you are contributing to NeMo Curator! Before you make a PR, be sure to read over this guide in detail.
This checklist ensures that NeMo Curator stays easy-to-use by both users and developers.
Not all steps are necessary for some contributions, so read the linked sections for more information about each item.

1. [Follow the general principles in your design](#general-principles)
1. [Write your code in the proper place](#repo-structure)
1. [Write examples and documentation for using your code](#examples-and-documentation)
1. [Format using the style guide](#python-style)
1. [Write unit tests](#unit-tests)
1. [Make a pull request](#pull-requests-pr-guidelines)

## General principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
1. **Robust**: make it hard for users to make mistakes.
1. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to be reused.
1. **Readable**: code should be easier to read.
1. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that NeMo Curator supports. Give credit and link back to the code.
1. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.

## Code Structure
The repository is home to flexible Python modules, sample scripts, tests, and more.
Here is a brief overview of where everything lives:
- [config](config/) - A collection of example configuration files for many of the curator's modules.
- [docs](docs/) - Walkthroughs and motivations for each of the modules.
- [examples](examples/) - Example scripts for how users may want to compose the curator.
- [nemo_curator](nemo_curator/) - The main home for all the NeMo Curator's Python APIs.
    - [modules](nemo_curator/modules) - Classes for the modules.
    - [filters](nemo_curator/filters) - Classes for the filters.
    - [utils](nemo_curator/utils) - Common utilities for file/network operations.
- [tests](tests/) - Unit tests for each module.

## Examples and Documentation
Examples provide an easy way for users to see how the curator works in action.
There should be at least one example per module in the curator.
They should be incredibly lightweight and rely on the core `nemo_curator` modules for their functionality.
Most should be designed for a user to get up and running on their local machines, but distributed examples are welcomed if it makes sense.
Python scripts should be the primary way to showcase your module.
Though, SLURM scripts or other cluster scripts should be included if there are special steps needed to run the module.

The documentation should complement each example by going through the motivation behind why a user would use each module.
It should include both an explanation of the module, and how it's used in its corresponding example.
The documentation should also cover potential pitfalls and performance considerations when running the module at scale.
The existing examples and documentation should serve as a good reference to what is expected.

## Python style
We use ``black`` as our style guide. To fix your format run `pip install pre-commit && pre-commit install && pre-commit run --all`.

1. Include docstrings for every class and method exposed to the user.
1. Avoid wild import: ``from X import *`` unless in ``X.py``, ``__all__`` is defined.
1. Minimize the use of ``**kwargs``.
1. ``RaiseError`` is preferred to ``assert``. Write: ```if X: raise Error``` instead of ```assert X```.
1. Classes are preferred to standalone methods.
1. Methods should be atomic. A method shouldn't be longer than 88 lines, e.g. can be fit into the computer screen without scrolling.
1. If a method has arguments that don't fit into one line, each argument should be in its own line for readability.
1. Add ``__init__.py`` for every folder.
1. F-strings are prefered to formatted strings.
1. Loggers are preferred to print.
1. Private functions (functions start with ``_``) shouldn't be called outside its host file.
1. If a comment lasts multiple lines, use ``'''`` instead of ``#``.

## Unit tests
Unit tests should be simple and fast.
Developers should be able to run them frequently while developing without any slowdown.
```
pytest
# If you don't have NVIDIA GPU do:
# pytest --cpu
```

## Pull Requests (PR) Guidelines

**Send your PRs to the `main` branch**

1) Make sure your PR does one thing. Have a clear answer to "What does this PR do?".
2) Read General Principles and style guide above
3) Make sure you sign your commits. E.g. use ``git commit -sS`` when committing.
    1) If you forget to do this, please follow the steps below to undo the commits and reapply the changes under a new (signed and signed-off) commit. Note: This will preserve your changes, but delete the git history of commits.
    ```bash
    git reset --soft HEAD~N
    git add <insert all files you want to include>
    git commit -sS -m "My commit message"
    git push --force
    ```
    Replace `N` in the first line with the number of commits you want to undo. To undo the latest commit, do `git reset --soft HEAD~1`.
4) Make sure all unittests finish successfully before sending PR ``pytest`` or (if your dev box does not have GPU) ``pytest --cpu`` from the root folder
5) Send your PR and request a review

Unit tests are expected to pass before merging into `main`.
Every release a new branch will be cut from `main`.

Full text of the DCO:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

## Whom should you ask for review:

Ryan Wolf (@ryantwolf), Ayush Dattagupta (@ayushdg), Vibhu Jawa (@VibhuJawa), or Sarah Yurick (@sarahyurick)

They may ask for other reviewers depending on the scope of the change. Your pull requests must pass all checks and peer-review before they can be merged.


Thank you for contributing to NeMo Curator!
