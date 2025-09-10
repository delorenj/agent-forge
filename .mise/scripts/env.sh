# Auto-exec script to bootstrap dev environment
export APP_NAME="MyApp"
export APP_VERSION="0.1.0"

# Guard against recursive execution
if [ -n "$HOOK_RUNNING" ]; then
  exit 0
fi
export HOOK_RUNNING=1

# Timestamp-based guard to prevent multiple executions within 5 seconds
HOOK_TIMESTAMP_FILE=".hook_timestamp"
CURRENT_TIME=$(date +%s)

if [ -f "$HOOK_TIMESTAMP_FILE" ]; then
  LAST_RUN=$(cat "$HOOK_TIMESTAMP_FILE" 2>/dev/null || echo "0")
  TIME_DIFF=$((CURRENT_TIME - LAST_RUN))
  if [ "$TIME_DIFF" -lt 5 ]; then
    echo "Hook ran recently, skipping to prevent double execution..."
    unset HOOK_RUNNING
    exit 0
  fi
fi

echo "$CURRENT_TIME" >"$HOOK_TIMESTAMP_FILE"

BANNER="
$APP_NAME $APP_VERSION
"
echo -e "$BANNER"
echo "Welcome to $APP_NAME development environment!"

# If $SKIP_HOOK is set, skip the rest of the script
if [ -n "$SKIP_HOOK" ]; then
  echo "SKIP_HOOK is set. Skipping the hook..."
  exit 0
fi

# Check if environment is already set up
check_env_status() {
  # Create a file to track if the environment has been set up in this session
  ENV_STATUS_FILE=".env_status"

  if [ -f "$ENV_STATUS_FILE" ]; then
    # Environment already set up in this session
    return 0
  fi
  return 1
}

# Mark environment as set up
mark_env_setup() {
  ENV_STATUS_FILE=".env_status"
  touch "$ENV_STATUS_FILE"
}

# Check and install mise if needed
check_and_install_mise() {
  if ! command -v mise &>/dev/null; then
    echo "mise is not installed. Installing mise..."
    curl https://mise.run | sh

    # Check if mise is available in the PATH after installation
    if ! command -v mise &>/dev/null; then
      echo "mise installed but not in PATH. Please restart your shell or source your profile."
      echo "Then rerun this script."
      return 1
    fi
    echo "mise installed successfully! Adding shims to zsh..."
    echo "mise installed successfully! To activate it, add the following to your ~/.zshrc (or equivalent for your shell):"
    echo 'eval \"$(mise activate zsh)\"'
  fi
  return 0
}

install_dependencies() {
  check_and_install_mise
  echo "Install all tool dependencies using mise..."
  mise install
  if ! command -v uv &>/dev/null; then
    echo "mise shims not initialized. Running 'eval \"\$(mise activate zsh)\"' to fix."
    eval "$(mise activate zsh)"
    echo "Attempting to install tool dependencies one more time..."
    mise install
    if ! command -v uv &>/dev/null; then
      echo "Uh oh... I'm out of ideas. Please ensure 'mise' is correctly activated in your shell."
      return 1
    fi
  fi
  return 0
}

# Setup Python virtual environment if needed
activate() {
  install_dependencies
  uv sync
  echo "Dev environment ready!"
  return 0
}

# Only run the setup logic in an interactive terminal
if [ -t 0 ]; then
  activate

  # Check if environment is already set up in this session
  if check_env_status; then
    echo "Environment already set up in this session. Skipping startup tasks."
    export HOOK_RUNNING=""
    exit 0
  fi
fi

# Reset the guard at the end
export HOOK_RUNNING=""
