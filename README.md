41118 AI Robotics Project
This repository is used for our group project for 41118 AI Robotics.
We are developing the project in a WSL/Linux environment and using GitHub with SSH authentication to collaborate, push, pull, commit, and manage code changes.
Repository:
```bash
git@github.com:elsayeddd/41118-AI-Robotics-Project.git
```
GitHub page:
```bash
https://github.com/elsayeddd/41118-AI-Robotics-Project
```
---
1. Before You Start
Each group member needs:
WSL installed
Git installed in WSL
VS Code installed
The VS Code WSL extension installed
A GitHub account
Access to this GitHub repository
Make sure you have accepted the GitHub repository invitation before trying to clone or push.
---
2. Open WSL
Open your WSL terminal, for example Ubuntu.
It is recommended to store the project inside the WSL file system, not directly inside the Windows `C:` drive.
A good location is:
```bash
/home/your-username/projects
```
---
3. Install Git in WSL
Check whether Git is already installed:
```bash
git --version
```
If Git is not installed, run:
```bash
sudo apt update
sudo apt install git
```
---
4. Set Up Your Git Name and Email
Each group member should set their own Git name and UTS student email.
Run:
```bash
git config --global user.name "First Name Last Name"
git config --global user.email "firstname.lastname@student.uts.edu.au"
```
Example:
```bash
git config --global user.name "John Smith"
git config --global user.email "john.smith@student.uts.edu.au"
```
> Note: The usual UTS student email format does **not** include an extra dot before the `@` symbol. Use `firstname.lastname@student.uts.edu.au` unless UTS specifically issued your email differently.
Check that your details were saved:
```bash
git config --global --list
```
You should see your name and email listed.
---
5. Set Up SSH for GitHub in WSL
GitHub does not accept normal account passwords when using Git in the terminal.
Instead, we will use SSH authentication.
SSH lets your WSL environment securely connect to GitHub without typing your GitHub password every time you push or pull.
---
6. Check for Existing SSH Keys
In WSL, run:
```bash
ls -al ~/.ssh
```
Look for files such as:
```bash
id_ed25519
id_ed25519.pub
```
If these files already exist, you may already have an SSH key.
If they do not exist, continue to the next step.
---
7. Generate a New SSH Key
Run the following command in WSL, replacing the email with your own UTS student email:
```bash
ssh-keygen -t ed25519 -C "firstname.lastname@student.uts.edu.au"
```
Example:
```bash
ssh-keygen -t ed25519 -C "john.smith@student.uts.edu.au"
```
When it asks where to save the key, press Enter to accept the default location:
```bash
/home/your-username/.ssh/id_ed25519
```
When it asks for a passphrase, you can either:
press Enter for no passphrase, or
enter a passphrase for extra security
For this group project, pressing Enter is usually simpler.
---
8. Start the SSH Agent
Run:
```bash
eval "$(ssh-agent -s)"
```
This starts the SSH agent, which manages your SSH keys.
---
9. Add Your SSH Key to the SSH Agent
Run:
```bash
ssh-add ~/.ssh/id_ed25519
```
If successful, you should see a message similar to:
```bash
Identity added: /home/your-username/.ssh/id_ed25519
```
---
10. Copy Your Public SSH Key
Run:
```bash
cat ~/.ssh/id_ed25519.pub
```
This will print a long line of text starting with something like:
```bash
ssh-ed25519
```
Copy the entire line.
Make sure you copy the `.pub` key, not the private key.
Do not share this file:
```bash
~/.ssh/id_ed25519
```
It is your private key.
Only share or upload this file:
```bash
~/.ssh/id_ed25519.pub
```
This is your public key.
---
11. Add the SSH Key to GitHub
Go to GitHub in your browser.
Then go to:
```text
GitHub → Profile picture → Settings → SSH and GPG keys → New SSH key
```
Use a clear title, for example:
```text
WSL Laptop
```
Paste the public key into the key box.
Then click:
```text
Add SSH key
```
---
12. Test the SSH Connection
Back in WSL, run:
```bash
ssh -T git@github.com
```
The first time, it may ask:
```bash
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```
Type:
```bash
yes
```
If it works, you should see something similar to:
```bash
Hi your-github-username! You've successfully authenticated, but GitHub does not provide shell access.
```
This means SSH is working correctly.
---
13. Clone the Repository Using SSH
Go to the location where you want to store the project:
```bash
cd ~
mkdir -p projects
cd projects
```
Clone the repository using the SSH link:
```bash
git clone git@github.com:elsayeddd/41118-AI-Robotics-Project.git
```
Enter the project folder:
```bash
cd 41118-AI-Robotics-Project
```
---
14. Open the Project in VS Code
From inside the project folder, run:
```bash
code .
```
This opens the repository in VS Code through WSL.
If `code .` does not work:
Open VS Code
Install the WSL extension
Reopen your WSL terminal
Try again:
```bash
code .
```
---
15. Basic Git Workflow
Before starting work, always pull the latest version:
```bash
git pull
```
Check which files have changed:
```bash
git status
```
Add your changed files:
```bash
git add .
```
Commit your changes:
```bash
git commit -m "Describe what you changed"
```
Push your changes to GitHub:
```bash
git push
```
The usual workflow is:
```bash
git pull
# edit files
git status
git add .
git commit -m "message"
git push
```
---
16. Recommended Collaboration Workflow
To avoid overwriting each other's work, each person should work on their own branch.
Create a new branch:
```bash
git checkout -b feature/your-name-or-task
```
Example:
```bash
git checkout -b feature/path-planning
```
After making changes:
```bash
git add .
git commit -m "Added path planning code"
git push -u origin feature/path-planning
```
Then go to GitHub and create a Pull Request to merge your branch into the main branch.
This is safer than everyone pushing directly to `main`.
---
17. Pulling the Latest Changes
Before starting work each time, run:
```bash
git pull
```
If you are working on a branch, first check which branch you are on:
```bash
git branch
```
Switch branches using:
```bash
git checkout branch-name
```
Example:
```bash
git checkout main
```
Then pull the latest changes:
```bash
git pull
```
---
18. Checking the Current Branch
To see what branch you are currently on:
```bash
git branch
```
The current branch will have a `*` next to it.
Example:
```bash
* main
  feature/path-planning
```
---
19. Common Git Commands
Command	Purpose
`git status`	Shows changed files
`git pull`	Downloads latest changes from GitHub
`git add .`	Stages all changed files
`git commit -m "message"`	Saves changes locally with a message
`git push`	Uploads changes to GitHub
`git branch`	Shows available branches
`git checkout branch-name`	Switches to another branch
`git checkout -b branch-name`	Creates and switches to a new branch
`ssh -T git@github.com`	Tests SSH connection to GitHub
---
20. Important Collaboration Rules
Always run `git pull` before starting work.
Do not push directly to `main` unless the group agrees.
Use branches for new features or individual tasks.
Write clear commit messages.
Communicate with the group before editing the same file.
Use Pull Requests where possible.
Check GitHub regularly for updated files, comments, and pull requests.
---
21. If You Already Cloned Using HTTPS
If your repository was cloned using HTTPS, you can change it to SSH.
Check the current remote:
```bash
git remote -v
```
If it shows something like:
```bash
https://github.com/elsayeddd/41118-AI-Robotics-Project.git
```
change it to SSH:
```bash
git remote set-url origin git@github.com:elsayeddd/41118-AI-Robotics-Project.git
```
Check again:
```bash
git remote -v
```
It should now show:
```bash
git@github.com:elsayeddd/41118-AI-Robotics-Project.git
```
Now you can push and pull using SSH:
```bash
git pull
git push
```
---
22. If You Already Have Project Files in WSL
If you already created files before cloning the repo, the easiest method is usually:
Clone the GitHub repo.
Copy your existing files into the cloned repo folder.
Commit and push them.
Example:
```bash
cd ~/projects
git clone git@github.com:elsayeddd/41118-AI-Robotics-Project.git
cd 41118-AI-Robotics-Project
```
Then copy your files into this folder.
After that:
```bash
git add .
git commit -m "Added initial project files"
git push
```
---
23. Troubleshooting
Permission denied publickey
If you see:
```bash
Permission denied (publickey).
```
Try the following:
```bash
ssh -T git@github.com
```
If SSH does not authenticate, check that:
you generated an SSH key in WSL
you added the public key to GitHub
you added the key to the SSH agent
you accepted the repository invitation
you are using the SSH clone link, not the HTTPS link
You can add the key again using:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```
---
Repository not found
If you see:
```bash
Repository not found.
```
Check that:
you accepted the GitHub invitation
you are logged into the correct GitHub account
the repository link is correct
you have permission to access the repository
---
Push rejected
This usually means someone else has pushed changes before you.
Run:
```bash
git pull
```
Then resolve any conflicts if needed, commit again, and push.
---
Merge conflict
A merge conflict happens when two people edit the same part of the same file.
Git will mark the conflict in the file.
Open the file, choose which version to keep, then run:
```bash
git add .
git commit -m "Resolved merge conflict"
git push
```
---
24. Summary Setup Commands
Each group member can use this as the main setup sequence:
```bash
sudo apt update
sudo apt install git

git config --global user.name "First Name Last Name"
git config --global user.email "firstname.lastname@student.uts.edu.au"

ssh-keygen -t ed25519 -C "firstname.lastname@student.uts.edu.au"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```
After copying the public key into GitHub, test SSH:
```bash
ssh -T git@github.com
```
Then clone the repository:
```bash
cd ~
mkdir -p projects
cd projects
git clone git@github.com:elsayeddd/41118-AI-Robotics-Project.git
cd 41118-AI-Robotics-Project
code .
```
When working on the project, use:
```bash
git pull
# make changes
git status
git add .
git commit -m "message"
git push
```
For proper collaboration, use branches and Pull Requests instead of everyone pushing directly to `main`.
