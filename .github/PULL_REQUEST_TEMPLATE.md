### Description

<!--  
Please insert a brief description of the Pull request here
-->

### Related issues or pull requests

<!--  
If applicable, please link related issues/pull request here. E.g.,   
Fixes #366
-->

### Pull Request Checklist

- [ ] Added a note about the modification or contribution to the `./docs/sources/CHANGELOG.md` file (if applicable)
- [ ] Added appropriate unit test functions in the `./mlearner/*/tests` directories (if applicable)
- [ ] Modify documentation in the corresponding Jupyter Notebook under `mlearner/docs/sources/` (if applicable)
- [ ] Ran `PYTHONPATH='.' pytest ./mlearner -sv` and make sure that all unit tests pass (for small modifications, it might be sufficient to only run the specific test file, e.g., `PYTHONPATH='.' pytest ./mlearner/classifier/tests/xxx.py -sv`)
- [ ] Checked for style issues by running `flake8 ./mlearner`


<!--NOTE  
Due to the improved GitHub UI, the squashing of commits is no longer necessary.
Please DO NOT SQUASH commits since they help with keeping track of the changes during the discussion).
For more information and instructions, please see http://jaisenbe58r.github.io/mlearner/contributing/  
-->
