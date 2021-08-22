#!/bin/bash
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

set -e

# Exec the specified command or fall back on bash
if [ $# -eq 0 ]; then
  cmd=("bash")
else
  cmd=("$@")
fi

run-hooks() {
  # Source scripts or run executable files in a directory
  if [[ ! -d "$1" ]]; then
    return
  fi
  echo "$0: running hooks in $1"
  for f in "$1/"*; do
    case "$f" in
    *.sh)
      echo "$0: running $f"
      source "$f"
      ;;
    *)
      if [[ -x "$f" ]]; then
        echo "$0: running $f"
        "$f"
      else
        echo "$0: ignoring $f"
      fi
      ;;
    esac
  done
  echo "$0: done running hooks in $1"
}

run-hooks /usr/local/bin/start-notebook.d

echo "Running as ${IMAGE_USER}"
if [[ "$USER_ID" == "$(id -u student)" && "$USER_GID" == "$(id -g student)" ]]; then
  # User is not attempting to override user/group via environment
  # variables, but they could still have overridden the uid/gid that
  # container runs as. Check that the user has an entry in the passwd
  # file and if not add an entry.
  STATUS=0 && whoami &>/dev/null || STATUS=$? && true
  if [[ "$STATUS" != "0" ]]; then
    if [[ -w /etc/passwd ]]; then
      echo "Adding passwd file entry for $(id -u)"
      cat /etc/passwd | sed -e "s/^student:/student:/" >/tmp/passwd
      echo "student:x:$(id -u):$(id -g):,,,:/home/student:/bin/bash" >>/tmp/passwd
      cat /tmp/passwd >/etc/passwd
      rm /tmp/passwd
    else
      echo 'Container must be run with group "root" to update passwd file'
    fi
  fi

  # Warn if the user isn't going to be able to write files to $HOME.
  if [[ ! -w /home/student ]]; then
    echo 'Container must be run with group "users" to update files'
  fi
else
  # Warn if looks like user want to override uid/gid but hasn't
  # run the container as root.
  if [[ ! -z "$USER_ID" && "$USER_ID" != "$(id -u)" ]]; then
    echo 'Container must be run as root to set $USER_ID'
  fi
  if [[ ! -z "$USER_GID" && "$USER_GID" != "$(id -g)" ]]; then
    echo 'Container must be run as root to set $USER_GID'
  fi
fi

# Warn if looks like user want to run in sudo mode but hasn't run
# the container as root.
if [[ "$GRANT_SUDO" == "1" || "$GRANT_SUDO" == 'yes' ]]; then
  echo 'Container must be run as root to grant sudo permissions'
fi

# Execute the command
run-hooks /usr/local/bin/before-notebook.d
echo "Executing the command: ${cmd[@]}"
exec "${cmd[@]}"
