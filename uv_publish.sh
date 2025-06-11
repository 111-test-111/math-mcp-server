# 设置环境变量
export UV_PUBLISH_TOKEN="pypi-token-here"
rm -rf dist/*
uv build
uv publish
