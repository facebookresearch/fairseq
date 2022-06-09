# Creating a New Release

In order to create a new release:

1. Navigate to the [Fairseq Workflows](https://github.com/facebookresearch/fairseq/actions) and find the one named _Fairseq Release_. 

2. Under _Run Workflow_ choose the branch `main` and for _Release Type_ enter either `major`, `minor`, or `patch`.  

3. A branch with the same name as the new version will be created where the `version.txt` file is updated. Merge those changes into `main`.

4. Make sure that a [new PYPI package](https://pypi.org/project/fairseq/) has been uploaded.

5. Make sure that a [new github release](https://github.com/facebookresearch/fairseq/releases) has been created.
