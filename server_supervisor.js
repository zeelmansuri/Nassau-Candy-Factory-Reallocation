const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const out = fs.createWriteStream(path.join(root, "outputs", "supervisor.log"), { flags: "a" });

function log(line) {
  out.write(`[${new Date().toISOString()}] ${line}\n`);
}

function start(name, command, args) {
  const child = spawn(command, args, {
    cwd: root,
    windowsHide: true,
    detached: false,
    stdio: ["ignore", "pipe", "pipe"],
  });
  log(`${name} pid=${child.pid}`);
  child.stdout.on("data", (chunk) => log(`${name} stdout: ${chunk.toString().trim()}`));
  child.stderr.on("data", (chunk) => log(`${name} stderr: ${chunk.toString().trim()}`));
  child.on("exit", (code, signal) => log(`${name} exited code=${code} signal=${signal}`));
  return child;
}

start("streamlit", "D:\\python\\python.exe", [
  "-m",
  "streamlit",
  "run",
  "app.py",
  "--server.port",
  "8501",
  "--server.headless",
  "true",
]);

setTimeout(() => {
  start("localtunnel", "cmd.exe", [
    "/c",
    "C:\\Program Files\\nodejs\\npx.cmd",
    "--yes",
    "localtunnel",
    "--port",
    "8501",
    "--local-host",
    "127.0.0.1",
    "--subdomain",
    "nassau-candy-opt-8501",
  ]);
}, 8000);

setInterval(() => log("heartbeat"), 30000);
