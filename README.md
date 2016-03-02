# ephys-analysis
scripts and utilities for processing electrophysiology data

# mini-hackathon

## goals
- pull together a repository of code for common ephys analyses
  - isolation quality
  - spike width
  - neuron location
  - rasters
  - PSTHs
  - behavior parsing
- ensure code is well documented
  - numpy docstrings (see https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
- everyone can learn how to branch/commit/pull-request/rebase
- maybe @MarvinT can even teach us how to do unit tests?

## approach

If you have code written already that works toward these ends, do the following...
see "for dummies" list below if needed
1. create a new branch of this repository
2. add your files to `ephys`
3. open a pull request for your branch

We'll hack on and merge branches as needed during the hackathon.

## approach (for dummies)
if you are not already familiar with this process you might enjoy these hopefully interpretable steps
(using the language of MAC OSx)
- Clone the repository on your local computer.
	- From the terminal command line on your local computer 
	- change to the directory you want this repository to live, 
	- RUN git clone [https link for repository]
- Branch the repository before making any changes (add files, modify files, etc)
	- from within the repository (so cd ../ephys-analysis)
	- RUN git checkout -b [whatever you want to name your branch]
- Add (file system language) files to "ephys" directory
	- copy your script that you want to contribute (using cp command in terminal OR dragging and dropping using Finder)
	- (or if you do not already have the code written, create a new file from withing ephys and write whatever script you want)
- "Add" (git language) - means something more specific than "copy"
	- RUN git add .
	- Adds the files in the local repository and stages them for commit. To unstage a file, use 'git reset HEAD YOUR-FILE'.
- commit these changes 
	- Commits the tracked changes and prepares them to be pushed to a remote repository. To remove this commit and modify the file, use 'git reset --soft HEAD~1' and commit and add the file again.
	- RUN git commit -m "whatever message you want to provide about what you did"
- push the changes to the repository [on github?] 
	- RUN git push origin [whatever you named your branch in step #2]
	- Pushes the changes in your local repository up to the remote repository you specified as the origin
- go online to github.com/gentnerlab/ephys-analysis/ to create a pull request for your branch
	- after you "pushed your branch" from your computer, your branch should show up in the list under the branches tab in this repository
	- CLICK on the "Pull Requests" tab
	- CLICK "New Pull Request"
	- CHOOSE "base: master" and "compare: [whatever you named your branch in step#2]"
- (?)if you make more changes to your branch after that.... create new pull request for same branch? does it automatically recognize new changes to that branch within the old pull request?


