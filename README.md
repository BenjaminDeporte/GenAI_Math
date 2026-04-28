WHAT IS THIS ALL ABOUT
----------------------

Generic code to test various ways of teaching LLMs math

Dataset : NuminaMath (see paper)

Tests:

1. Baseline with API calls to frontier models
2. SFT on opensource model
3. GRPO on opensource model


ENVIRONMENT VARIABLES
---------------------

NB : environment variables are supposed to be in a .env file, and loaded automatically in VS Code:
Open VS Code settings (Cmd+, on Mac, or Ctrl+, on Windows/Linux)
Search for python.terminal.useEnvFile
Check the box to enable it
Or add this to your .vscode/settings.json:
"python.terminal.useEnvFile": true
After enabling, new terminal sessions will automatically load variables from your .env file. 