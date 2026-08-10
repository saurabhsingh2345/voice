// Local Voice Agent desktop shell.
//
// The whole UI already exists as a local web app, so this shell does two jobs:
// start the Python server as a child process, and show its page in a native
// window. Everything still runs on this machine; the window points at
// 127.0.0.1 and nothing is fetched from the internet.
//
// The server is a child process rather than a bundled runtime, which means the
// app currently requires the project checkout and `uv` to be present. Embedding
// a Python runtime is a separate job -- and one with a licensing constraint:
// num2words is LGPL, so the dependency must stay replaceable rather than being
// frozen into one opaque archive.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

const HOST: &str = "127.0.0.1";
const PORT: u16 = 8823;
/// Model loading is lazy, so the server binds quickly. This is generous anyway
/// because a cold `uv run` may resolve the environment first.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(90);

/// Holds the server process so it can be killed when the window closes.
/// Without this the Python process outlives the app and keeps the port bound.
struct ServerProcess(Mutex<Option<Child>>);

fn project_root() -> Option<PathBuf> {
    // In a bundled .app the binary sits in Contents/MacOS, so walk up looking
    // for the project's pyproject.toml. During `tauri dev` the CWD is already
    // inside the project.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        candidates.extend(exe.ancestors().map(PathBuf::from));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.extend(cwd.ancestors().map(PathBuf::from));
    }
    candidates
        .into_iter()
        .find(|dir| dir.join("pyproject.toml").is_file())
}

fn port_is_open() -> bool {
    TcpStream::connect_timeout(
        &format!("{HOST}:{PORT}").parse().expect("valid socket address"),
        Duration::from_millis(300),
    )
    .is_ok()
}

fn spawn_server(root: &PathBuf) -> std::io::Result<Child> {
    // `uv run` resolves the project environment itself, so the shell does not
    // need to know where the venv lives.
    Command::new("uv")
        .args(["run", "voice-web"])
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
}

fn wait_for_server() -> bool {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    while Instant::now() < deadline {
        if port_is_open() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    false
}

fn main() {
    let app = tauri::Builder::default()
        .manage(ServerProcess(Mutex::new(None)))
        .setup(|app| {
            // Reuse a server the user already started rather than fighting it
            // for the port.
            let already_running = port_is_open();

            if !already_running {
                match project_root() {
                    Some(root) => match spawn_server(&root) {
                        Ok(child) => {
                            *app.state::<ServerProcess>().0.lock().unwrap() = Some(child);
                        }
                        Err(err) => {
                            eprintln!("could not start the voice server: {err}");
                        }
                    },
                    None => eprintln!("could not locate the project (no pyproject.toml above the binary)"),
                }
            }

            let url = if wait_for_server() {
                format!("http://{HOST}:{PORT}")
                    .parse()
                    .map(WebviewUrl::External)
                    .unwrap_or(WebviewUrl::App("index.html".into()))
            } else {
                // Fall back to the bundled page, which explains what went wrong
                // instead of showing a blank window.
                WebviewUrl::App("index.html".into())
            };

            WebviewWindowBuilder::new(app, "main", url)
                .title("Local Voice Agent")
                .inner_size(860.0, 940.0)
                .min_inner_size(520.0, 600.0)
                .build()?;

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the app");

    app.run(|handle, event| {
        if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
            if let Some(mut child) = handle.state::<ServerProcess>().0.lock().unwrap().take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    });
}
