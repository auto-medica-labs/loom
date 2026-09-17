---
description: Use this skill when tmux is needed for a persistent shell, interactive process, background job, parallel jobs, delayed output capture, or a user-requested long-running service. Use it whenever tmux is mentioned or when a command must continue after the invoking tool returns. Clean up temporary sessions when their jobs finish; leave intentionally persistent services running.
metadata:
    github-path: skills/tmux
    github-ref: refs/heads/main
    github-repo: https://github.com/dheerapat/skills
    github-tree-sha: 8cfd87555acdcec6fd82b97b13ad23bc708c0fde
name: tmux
---
# tmux Skill

A structured workflow for creating tmux sessions, executing commands, capturing output, and cleaning up — designed for agents running tasks that need persistent or background shells.

______________________________________________________________________

## Core Workflow

### 1. Create a Session

Create a **named session** so you can reference it reliably later. Prefer a unique, task-specific name to avoid terminating or reusing someone else's session.

```bash
tmux new-session -d -s <session-name>
```

- `-d` starts it detached (in the background)
- `-s <session-name>` assigns a name (use short, descriptive names like `build`, `server`, `task-1`)
- Default window/pane is `<session-name>:0.0`

**Check if a session already exists before creating:**

```bash
tmux has-session -t <session-name> 2>/dev/null
# Exit status 0 means it exists; nonzero means it does not.
```

______________________________________________________________________

### 2. Send Keys (Run Commands)

Use `send-keys` to type a command into a session pane, followed by `C-m` (Enter):

```bash
tmux send-keys -t <session-name> '<command>' C-m
```

**Examples:**

```bash
# Run a build
tmux send-keys -t build 'npm run build' C-m

# Start a server
tmux send-keys -t server 'python -m http.server 8080' C-m

# Chain commands
tmux send-keys -t task-1 'cd /tmp && ls -la' C-m
```

**Tips:**

- Always include `C-m` (or `Enter`) at the end to execute the command.
- The shell running `tmux send-keys` parses quotes, variables, command substitutions, and backslashes before tmux receives them. Quote the command appropriately, and do not interpolate untrusted input into a shell command.
- For complex or multiline commands, write a temporary script or use a carefully quoted `bash -lc` invocation instead of relying on nested shell quoting.
- To send a literal `Enter` without a command: `tmux send-keys -t <session-name> C-m`
- To send `Ctrl+C` to stop a process: `tmux send-keys -t <session-name> C-c`

______________________________________________________________________

### 3. Capture Output

Use `capture-pane` to retrieve visible terminal content from the pane:

```bash
tmux capture-pane -t <session-name> -p
```

- `-p` prints output to stdout (instead of a paste buffer)
- Returns the current visible contents of the pane

**Capture including scrollback history:**

```bash
tmux capture-pane -t <session-name> -p -S -3000
```

- `-S -3000` goes back up to 3000 lines of history (adjust as needed)

**Save output to a file:**

```bash
tmux capture-pane -t <session-name> -p -S -3000 > /tmp/<session-name>-output.txt
```

______________________________________________________________________

### 4. Wait for Commands to Finish

For commands that take time, poll until they complete:

```bash
# Simple sleep-and-check loop. Use a unique completion marker.
for i in $(seq 1 30); do
  sleep 2
  output=$(tmux capture-pane -t <session-name> -p -S -500)
  echo "$output" | grep -Fq '<done-marker>' && break
done
```

Replace `<done-marker>` with a string that appears when the command finishes (e.g., `"Build complete"`, `"$ "` for a shell prompt, `"error"`, `"success"`).

**Better pattern — redirect output to a file and tail it:**

```bash
# Send command with output redirected and record its exit status.
tmux send-keys -t <session-name> '<command> > /tmp/<unique-output>.txt 2>&1; status=$?; printf "__EXIT__%s\\n" "$status" >> /tmp/<unique-output>.txt' C-m

# Poll the file
for i in $(seq 1 60); do
  sleep 1
  grep -Fq '__EXIT__' /tmp/<unique-output>.txt 2>/dev/null && break
done

cat /tmp/<unique-output>.txt
```

______________________________________________________________________

### 5. Kill (Remove) the Session

Clean up temporary sessions when the job is done:

```bash
tmux kill-session -t <session-name>
```

Do not use `tmux kill-server` in a shared environment: it terminates every tmux session, including unrelated user work.

______________________________________________________________________

## Full Example: Run a Script and Capture Output

```bash
# 1. Create session
tmux new-session -d -s myjob

# 2. Run command, redirect output and record its exit status
tmux send-keys -t myjob 'bash /tmp/myscript.sh > /tmp/myjob-out.txt 2>&1; status=$?; printf "__EXIT__%s\\n" "$status" >> /tmp/myjob-out.txt' C-m

# 3. Wait for completion (up to 60s)
for i in $(seq 1 60); do
  sleep 1
  grep -Fq "__EXIT__" /tmp/myjob-out.txt 2>/dev/null && break
done

# 4. Capture and display output
cat /tmp/myjob-out.txt

# 5. Kill session
tmux kill-session -t myjob 2>/dev/null || true
```

______________________________________________________________________

## Parallel Jobs Pattern

To run multiple tasks in parallel, create one session per task:

```bash
tmux new-session -d -s job1
tmux new-session -d -s job2

tmux send-keys -t job1 'task1.sh > /tmp/job1.txt 2>&1; status=$?; printf "__EXIT__%s\\n" "$status" >> /tmp/job1.txt' C-m
tmux send-keys -t job2 'task2.sh > /tmp/job2.txt 2>&1; status=$?; printf "__EXIT__%s\\n" "$status" >> /tmp/job2.txt' C-m

# Wait for both
for i in $(seq 1 60); do
  sleep 1
  j1=0
  j2=0
  grep -Fq '__EXIT__' /tmp/job1.txt 2>/dev/null && j1=1
  grep -Fq '__EXIT__' /tmp/job2.txt 2>/dev/null && j2=1
  [ "$j1" -ge 1 ] && [ "$j2" -ge 1 ] && break
done

cat /tmp/job1.txt
cat /tmp/job2.txt

tmux kill-session -t job1
tmux kill-session -t job2
```

______________________________________________________________________

## Useful Diagnostic Commands

```bash
# List all active sessions
tmux ls

# Show what's currently visible in a pane
tmux capture-pane -t <session-name> -p

# Check if session exists
tmux has-session -t <session-name> 2>/dev/null && echo "exists"

# List windows in a session
tmux list-windows -t <session-name>
```

______________________________________________________________________

## Common Pitfalls

| Problem                          | Fix                                                                                                 |
| -------------------------------- | --------------------------------------------------------------------------------------------------- |
| Command not running              | Check you included `C-m` or `Enter` in `send-keys`                                                  |
| Output is empty                  | Add a short `sleep 1` before `capture-pane`; the command may not have started yet                   |
| Session name conflict            | Use `tmux has-session` to check first, or use unique names                                          |
| Special chars in command         | Wrap the command string in single quotes, or escape `$`, `"`, `` ` ``                               |
| Process still running after kill | Check for child processes; `kill-session` ends the pane shell but may not stop detached descendants |

______________________________________________________________________

## Cleanup Reminder

**Kill temporary sessions after use.** Intentionally persistent services are the exception; document their session name and lifecycle.

```bash
tmux kill-session -t <session-name>
```

If you created multiple sessions, kill each one explicitly. Avoid `tmux kill-server`, which wipes all sessions for the tmux server.
