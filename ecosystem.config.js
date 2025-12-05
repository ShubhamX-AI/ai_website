module.exports = {
  apps: [
    {
      name: "ai_website",
      script: "server_run.py",
      interpreter: "./.venv/bin/python",
      env: {
        PORT: 8000
      }
    }
  ]
}