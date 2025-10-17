#!/bin/bash
# Basic secret scan (regex only) - DO NOT PRINT SECRET CONTENTS

echo '{
  "secret_patterns": [
    "password.*=.*[a-zA-Z0-9]+",
    "api_key.*=.*[a-zA-Z0-9]+", 
    "secret.*=.*[a-zA-Z0-9]+",
    "token.*=.*[a-zA-Z0-9]+",
    "key.*=.*[a-zA-Z0-9]+"
  ],
  "files_with_potential_secrets": []
}'

# Find files with potential secrets (just count, don't print content)
echo "Files with potential secrets:"
find . -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.md" | \
  xargs grep -l -E "(password|api_key|secret|token|key).*=" 2>/dev/null | \
  wc -l
