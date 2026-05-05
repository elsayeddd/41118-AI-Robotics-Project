# 41118 AI Robotics Project

This repository is used for our group project for **41118 AI Robotics**.  
We are developing the project in a **WSL/Linux environment** and using **GitHub** to collaborate, push, pull, commit, and manage code changes.

Repository link:

```bash
https://github.com/elsayeddd/41118-AI-Robotics-Project
```

---

## 1. Setting Up the Repository in WSL

Each group member should clone the repository into their own WSL environment.

Open your WSL terminal and run:

```bash
cd ~
mkdir projects
cd projects
git clone https://github.com/elsayeddd/41118-AI-Robotics-Project.git
cd 41118-AI-Robotics-Project
```

This creates a local copy of the GitHub repository inside WSL.

---

## 2. Installing Git in WSL

Check whether Git is installed:

```bash
git --version
```

If Git is not installed, run:

```bash
sudo apt update
sudo apt install git
```

---

## 3. Setting Up Your Git Identity

Each group member should set their Git name and email once.

Use your own name and UTS student email:

```bash
git config --global user.name "First Name Last Name"
git config --global user.email "firstname.lastname@student.uts.edu.au"
```

Example:

```bash
git config --global user.name "John Smith"
git config --global user.email "john.smith@student.uts.edu.au"
```

Make sure the email matches the email connected to your GitHub account.

> Note: The usual UTS student email format does **not** include an extra dot before the `@` symbol. Use `firstname.lastname@student.uts.edu.au` unless UTS specifically issued your email differently.

---

## 4. Opening the Project in VS Code

From inside the project folder, run:

```bash
code .
```

This opens the repository in VS Code using WSL.

If this does not work, install the **WSL extension** in VS Code and try again.

---

## 5. Basic Git Workflow

Before making changes, always pull the latest version:

```bash
git pull
```

After editing files, check what has changed:

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

The normal workflow is:

```bash
git pull
# edit files
git status
git add .
git commit -m "message"
git push
```

---

## 6. Recommended Collaboration Workflow

To avoid overwriting each other’s work, each person should work on their own branch.

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

Then go to GitHub and create a **Pull Request** to merge your branch into the main branch.

This is safer than everyone pushing directly to `main`.

---

## 7. Pulling the Latest Changes

Before starting work each time, run:

```bash
git pull
```

If you are working on a branch, make sure you are on the correct branch first:

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

---

## 8. Checking the Current Branch

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

## 9. Common Git Commands

| Command | Purpose |
|---|---|
| `git status` | Shows changed files |
| `git pull` | Downloads latest changes from GitHub |
| `git add .` | Stages all changed files |
| `git commit -m "message"` | Saves changes locally with a message |
| `git push` | Uploads changes to GitHub |
| `git branch` | Shows available branches |
| `git checkout branch-name` | Switches to another branch |
| `git checkout -b branch-name` | Creates and switches to a new branch |

---

## 10. Important Collaboration Rules

- Always run `git pull` before starting work.
- Do not push directly to `main` unless the group agrees.
- Use branches for new features or individual tasks.
- Write clear commit messages.
- Communicate with the group before editing the same file.
- Use Pull Requests where possible.
- Check GitHub regularly for updated files and comments.

---

## 11. If You Already Have Project Files in WSL

If you already created files before cloning the repo, the easiest method is usually:

1. Clone the GitHub repo.
2. Copy your existing files into the cloned repo folder.
3. Commit and push them.

Example:

```bash
cd ~/projects
git clone https://github.com/elsayeddd/41118-AI-Robotics-Project.git
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

## 12. Troubleshooting

### Permission denied when pushing

You may need to log in to GitHub or set up authentication.

If using HTTPS, GitHub may ask you to sign in using a personal access token instead of your password.

Alternatively, you can set up SSH keys for GitHub.

---

### Push rejected

This usually means someone else has pushed changes before you.

Run:

```bash
git pull
```

Then resolve any conflicts if needed, commit again, and push.

---

### Merge conflict

A merge conflict happens when two people edit the same part of the same file.

Git will mark the conflict in the file. Open the file, choose which version to keep, then run:

```bash
git add .
git commit -m "Resolved merge conflict"
git push
```

---

## 13. Summary Setup Commands

Each group member should run:

```bash
cd ~
mkdir projects
cd projects
git clone https://github.com/elsayeddd/41118-AI-Robotics-Project.git
cd 41118-AI-Robotics-Project
code .
```

Then use this workflow when working:

```bash
git pull
# make changes
git status
git add .
git commit -m "message"
git push
```

For proper collaboration, use branches and Pull Requests instead of everyone pushing directly to `main`.
